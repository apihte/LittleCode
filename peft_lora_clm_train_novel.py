import gc
import os
import sys
import threading
import json
import pandas as pd

import psutil
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model


# 计算两个字符串之间的Levenshtein距离，用于评估预测结果与真实标签之间的相似性
def levenshtein_distance(str1, str2):
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = list(range(num_cols))
    for i in range(1, num_rows):
        prev = dp_matrix[0]
        dp_matrix[0] = i
        for j in range(1, num_cols):
            temp = dp_matrix[j]
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[j] = prev
            else:
                dp_matrix[j] = min(prev, dp_matrix[j], dp_matrix[j - 1]) + 1
            prev = temp
    return dp_matrix[num_cols - 1]

# 找到与预测结果最接近的标签
def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]

# 将字节转换为兆字节
def b2mb(x):
    return int(x / 2**20)

# 内存监控工具类，用于监控训练过程中的内存使用情况
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()  # 垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.reset_peak_memory_stats()  # 重置最大内存分配计数器
        self.begin = torch.cuda.memory_allocated()  # 获取当前CUDA内存分配
        self.process = psutil.Process()  # 获取当前进程

        self.cpu_begin = self.cpu_mem_used()  # 获取CPU内存使用情况
        self.peak_monitoring = True  # 开始监控峰值内存使用情况
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)  # 创建监控线程
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss  # 获取当前进程的常驻集大小（RSS）内存

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)  # 更新CPU内存峰值
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()  # 再次进行垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存
        self.end = torch.cuda.memory_allocated()  # 获取当前CUDA内存分配
        self.peak = torch.cuda.max_memory_allocated()  # 获取CUDA最大内存分配
        self.used = b2mb(self.end - self.begin)  # 计算使用的CUDA内存
        self.peaked = b2mb(self.peak - self.begin)  # 计算峰值CUDA内存

        self.cpu_end = self.cpu_mem_used()  # 获取当前CPU内存使用
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)  # 计算使用的CPU内存
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)  # 计算峰值CPU内存

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["instruction"] + " " + item["input"]
        output_text = item["output"]
        inputs = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(output_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return {k: v.squeeze() for k, v in inputs.items()}

def main():
    accelerator = Accelerator()  # 初始化Accelerator以支持分布式训练
    model_name_or_path = "E:/workspace/other/Llama3-8B-Chinese-Chat"  # 模型路径
    data_file = "E:/workspace/other/peft/ought/raft/novel.json"  # 数据文件路径
    dataset_name = "novel"  # 数据集名称
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)  # PEFT配置
    text_column = "Tweet text"  # 文本列
    label_column = "text_label"  # 标签列
    lr = 3e-3  # 学习率
    num_epochs = 20  # 训练轮数
    batch_size = 8  # 批大小
    seed = 42  # 随机种子
    max_length = 512  # 最大序列长度
    do_test = True  # 是否进行测试
    set_seed(seed)  # 设置随机种子

    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # 加载分词器

    dataset = CustomDataset(data, tokenizer, max_length)  # 创建自定义数据集

    # 划分数据集为训练集、验证集和测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    print(next(iter(train_dataloader)))  # 打印第一个训练批次

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)  # 加载模型
    model = get_peft_model(model, peft_config)  # 应用PEFT配置
    model.print_trainable_parameters()  # 打印可训练参数

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # 创建优化器

    # 创建学习率调度器
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # 准备模型和数据加载器
    model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)  # 打印模型

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    # 训练循环
    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # 打印GPU和CPU内存使用情况
        accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        # 验证循环
        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(val_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                    )
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # 打印GPU和CPU内存使用情况
        accelerator.print(f"GPU Memory before entering the eval : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the eval (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the eval (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the eval (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the eval : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the eval (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the eval (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the eval (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )

        # 计算验证集上的准确率
        correct = 0
        total = 0
        for pred, true in zip(eval_preds, val_dataset):
            if pred.strip() == true['output'].strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")

    # 测试循环
    if do_test:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # 假设classes是数据集中所有唯一的output值
        classes = list(set([item['output'] for item in data]))

        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = pd.DataFrame([item for item in data])
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df['predicted_output'] = test_preds_cleaned
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[['instruction', 'input', 'predicted_output']].sample(20))

        pred_df = test_df[["ID", 'predicted_output']]
        pred_df.columns = ["ID", "Label"]

        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
    peft_model_id = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
        "/", "_"
    )
    model.save_pretrained(peft_model_id)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
