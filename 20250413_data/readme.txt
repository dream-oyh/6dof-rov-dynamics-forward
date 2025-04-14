该组数据由Overactuated-ROV-Simulation-oyh项目中产生，由程序控制仿真器多次仿真，通过位置接口控制机器人在水平面运动，只取了前3000个轨迹点，参考轨迹的xyz值满足：
t = 0.01:0.01:0.5;
t = t';
x = 5*sin(t);
y = 7*sin(2*t);
z = zeros(50,1);
yaw = 5*cos(t);
posedata中前六列是xyz, roll, pitch, yaw六列位置数据，后六列是速度数据，uvwpqr
thrustdata中是XYZKMN外力和外力矩