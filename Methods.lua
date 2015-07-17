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
