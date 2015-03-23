
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

	local flag = math.random(1, total)

	for key, _ in pairs(tmptbl) do
		if flag >= tmptbl[key][1] and flag <= tmptbl[key][2] then
			return key
		end
	end
end