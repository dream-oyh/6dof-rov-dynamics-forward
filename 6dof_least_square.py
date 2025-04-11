import datetime
import os

import numpy as np

from robot import MOTOR, ROV
from plot import plot
pose_dir = "matlab_data/pose_data.csv"
thruster_dir = "matlab_data/thrust_data.csv"

bluerov = ROV(pose_dir)
motor = MOTOR(thruster_dir)

tau = motor.tau
u, v, w = bluerov.vel_b[:-1, 0], bluerov.vel_b[:-1, 1], bluerov.vel_b[:-1, 2]
p, q, r = bluerov.vel_b[:-1, 3], bluerov.vel_b[:-1, 4], bluerov.vel_b[:-1, 5]
X, Y, Z, K, M, N = (
    tau[:-1, 0],
    tau[:-1, 1],
    tau[:-1, 2],
    tau[:-1, 3],
    tau[:-1, 4],
    tau[:-1, 5],
)

m = bluerov.mass
J_x = bluerov.J_x
J_y = bluerov.J_y
J_z = bluerov.J_z

u_dot, v_dot, w_dot = bluerov.acc[:, 0], bluerov.acc[:, 1], bluerov.acc[:, 2]
p_dot, q_dot, r_dot = bluerov.acc[:, 3], bluerov.acc[:, 4], bluerov.acc[:, 5]

F = np.zeros((6 * len(bluerov.timestamps), 1))
A = np.zeros((6 * len(bluerov.timestamps), 18))

for i in range(len(bluerov.timestamps) - 1):
    F[i * 6] = X[i] - m * u_dot[i] - (w[i] * q[i] - v[i] * r[i]) * m
    F[i * 6 + 1] = Y[i] - m * v_dot[i] - (u[i] * r[i] - w[i] * p[i]) * m
    F[i * 6 + 2] = Z[i] - m * w_dot[i] - (v[i] * p[i] - u[i] * q[i]) * m
    F[i * 6 + 3] = K[i] - J_x * p_dot[i] - (J_z - J_y) * q[i] * r[i]
    F[i * 6 + 4] = M[i] - J_y * q_dot[i] - (J_x - J_z) * p[i] * r[i]
    F[i * 6 + 5] = N[i] - J_z * r_dot[i] - (J_y - J_x) * p[i] * q[i]

    A[i * 6, 0:6] = [-u_dot[i], v[i] * r[i], -w[i] * q[i], 0, 0, 0]
    A[i * 6 + 1, 0:6] = [-u[i] * r[i], -v_dot[i], w[i] * p[i], 0, 0, 0]
    A[i * 6 + 2, 0:6] = [u[i] * q[i], -v[i] * p[i], -w_dot[i], 0, 0, 0]
    A[i * 6 + 3, 0:6] = [
        0,
        v[i] * w[i],
        -v[i] * w[i],
        -p_dot[i],
        q[i] * r[i],
        -q[i] * r[i],
    ]
    A[i * 6 + 4, 0:6] = [
        -u[i] * w[i],
        0,
        u[i] * w[i],
        -r[i] * p[i],
        -q_dot[i],
        r[i] * p[i],
    ]
    A[i * 6 + 5, 0:6] = [
        u[i] * v[i],
        -u[i] * v[i],
        0,
        p[i] * q[i],
        -p[i] * q[i],
        -r_dot[i],
    ]

    vel = -np.diag([u[i], v[i], w[i], p[i], q[i], r[i]])
    abs_vel = -np.diag(
        [
            u[i] * np.abs(u[i]),
            v[i] * np.abs(v[i]),
            w[i] * np.abs(w[i]),
            p[i] * np.abs(p[i]),
            q[i] * np.abs(q[i]),
            r[i] * np.abs(r[i]),
        ]
    )
    A[i * 6 : i * 6 + 6, :] = np.concatenate(
        [A[i * 6 : i * 6 + 6, 0:6], vel, abs_vel], axis=1
    )

theta = np.linalg.lstsq(A, F, rcond=None)[0]
theta = np.squeeze(theta)

output_result_file = "results/identify_results.md"

os.makedirs(os.path.dirname(output_result_file), exist_ok=True)
with open(output_result_file, "w", encoding="utf-8") as f:
    result = f"""
# {datetime.date.today()} identify results

identified by least square method

|参数 | 预测值 | 真实值 |
|:---:|:---:|:---:|
|X_udot|{theta[0]:.4f}|{bluerov.X_udot}|
|Y_vdot|{theta[1]:.4f}|{bluerov.Y_vdot}|
|Z_wdot|{theta[2]:.4f}|{bluerov.Z_wdot}|
|K_pdot|{theta[3]:.4f}|{bluerov.K_pdot}|
|M_qdot|{theta[4]:.4f}|{bluerov.M_qdot}|
|N_rdot|{theta[5]:.4f}|{bluerov.N_rdot}|
|X_u|{theta[6]:.4f}|{bluerov.X_u}|
|Y_v|{theta[7]:.4f}|{bluerov.Y_v}|
|Z_w|{theta[8]:.4f}|{bluerov.Z_w}|
|K_p|{theta[9]:.4f}|{bluerov.K_p}|
|M_q|{theta[10]:.4f}|{bluerov.M_q}|
|N_r|{theta[11]:.4f}|{bluerov.N_r}|
|X_uu|{theta[12]:.4f}|{bluerov.X_uu}|
|Y_vv|{theta[13]:.4f}|{bluerov.Y_vv}|
|Z_ww|{theta[14]:.4f}|{bluerov.Z_ww}|
|K_pp|{theta[15]:.4f}|{bluerov.K_pp}|
|M_qq|{theta[16]:.4f}|{bluerov.M_qq}|
|N_rr|{theta[17]:.4f}|{bluerov.N_rr}|
"""
    f.write(result)