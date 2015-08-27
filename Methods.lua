---------------------------------------------->>>
--Random
math.randomseed(tonumber(tostring(os.time()):reverse():sub(1, 6)))

---------------------------------------------->>>
--Random
function DoRandom(Min, Max)
	local szMin = tostring(Min)
	local szMax = tostring(Max)
	local subBit = math.max(string.len(szMin) - (string.find(szMin, "%.") or 1), string.len(szMax) - (string.find(szMax, "%.") or 1))
	print("subBit", subBit, string.len(szMin), "|", string.find(szMin, "."), "|", string.len(szMax), "|", string.find(szMax, "."))
	return math.random(Min*10^subBit, Max*10^subBit) / 10^subBit
end

---------------------------------------------->>>
--WeightRandom
local function WeightRandom(tbl)
	local st = 0
	local total = 0
	local tmptbl = {}

	for key, val in pairs(tbl) do
		tmptbl[key] = val
	end

	for key, val in pairs(tmptbl) do
		st = total
		total = total + val
		tmptbl[key] = {st, total}
	end

	local flag = DoRandom(1, total)

	for key, _ in pairs(tmptbl) do
		if flag >= tmptbl[key][1] and flag <= tmptbl[key][2] then
			return key
		end
	end
end

---------------------------------------------->>>
--Fibonacci
local function Fibonacci(uNumSum)
	local uBaseVal = 0
	local uAddVal = 1
	local FibonacciFunc
	FibonacciFunc = function(uNumSum)
		print(uBaseVal)
		if uNumSum == 0 then
			return
		end

		local uTmpVal = uBaseVal
		uBaseVal = uBaseVal + uAddVal
		uAddVal = uTmpVal
		uNumSum = uNumSum - 1
		return FibonacciFunc(uNumSum)
	end

	FibonacciFunc(uNumSum)
end

---------------------------------------------->>>
--QuickSort
function QuickSort(tSrcTbl)
	if #tSrcTbl == 0 then
		return
	end
	local QuickSortFunc
	QuickSortFunc = function(tSrcTbl, uLeft, uRight)
		if uLeft < uRight then
			local i = uLeft
			local j = uRight
			local uFlag = tSrcTbl[uLeft]
			while i < j do
				while i < j and tSrcTbl[j] >= uFlag do
					j = j - 1
				end
				while i < j and tSrcTbl[i] <= uFlag do
					i = i + 1
				end
				if i < j then
					local uTmp = tSrcTbl[i]
					tSrcTbl[i] = tSrcTbl[j]
					tSrcTbl[j] = uTmp
				end
			end
			tSrcTbl[uLeft] = tSrcTbl[i]
			tSrcTbl[i] = uFlag
			QuickSortFunc(tSrcTbl, uLeft, i - 1)
			QuickSortFunc(tSrcTbl, i + 1, uRight)
		end
	end

	QuickSortFunc(tSrcTbl, 1, #tSrcTbl)
end

---------------------------------------------->>>
--BubbleSort
function BubbleSort(tSrcTbl)
	if #tSrcTbl == 0 then
		return
	end
	for i = 1, #tSrcTbl do
		for j = i + 1, #tSrcTbl do
			if tSrcTbl[i] > tSrcTbl[j] then
				local uTmp = tSrcTbl[i]
				tSrcTbl[i] = tSrcTbl[j]
				tSrcTbl[j] = uTmp
			end
		end
	end
end