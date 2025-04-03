import pandas as pd

import robot
from plot import plot

pose_dir = "20250326_data/pose_data_3d.csv"
thrust_dir = "20250326_data/thrust_data_3d.csv"
# thrust_dir = "thrust_data.csv"

bluerov = robot.ROV(pose_dir)
motor = robot.MOTOR(thrust_dir)
motor.visualize_coordinate_system()
v_pred = bluerov.dynamics(motor)
v_true = bluerov.vel_b
plot(v_pred, v_true)


# tau_real = bluerov.dynamics(motor)
# tau_pred = motor.tau
# plot(tau_pred,tau_real)
