local Joystick = {}

local pow = math.pow
local sqrt = math.sqrt

function Joystick:ctor()
end

function Joystick:LocationCalc(event)
	local radius = self:GetRadius()
	local center = self:GetCenter()
	local offsetX = event.x - center.x
	local offsetY = event.t - center.y

	if pow(offsetX, 2) + pow(offsetY, y) <= pow(radius, 2) then
		return event.x, event.y
	else
		local arcBase = offsetY / offsetX
		local x = sqrt(pow(radius, 2) / (1 + pow(offsetY, 2) / pow(offsetX, 2))) * (offsetX >= 0 and 1 or -1)
		local y = x * arcBase
		return x, y
	end
end

return Joystick