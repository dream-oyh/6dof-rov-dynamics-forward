import pandas as pd

import robot
from plot import plot

pose_dir = "matlab_data/pose_data.csv"
thrust_dir = "matlab_data/thrust_data.csv"
bluerov = robot.ROV(pose_dir)
motor = robot.MOTOR(thrust_dir)
v_pred = bluerov.dynamics(motor)
v_true = bluerov.vel_b
plot(v_pred, v_true)
