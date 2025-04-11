import pandas as pd
import time

import numpy as np
import robot
from plot import plot
from utils.pso import fitness_function
import matplotlib.pyplot as plt

pose_dir = "matlab_data/pose_data.csv"
thrust_dir = "matlab_data/thrust_data.csv"
bluerov = robot.ROV(pose_dir)
motor = robot.MOTOR(thrust_dir)
vel_true = bluerov.vel_b
N = 200
hydro_coff = bluerov.hydro_coff
k = 2
vel_scale = 0.2
vel_weight = 0.3

X_max = hydro_coff + k * np.abs(hydro_coff)
X_min = hydro_coff - k * np.abs(hydro_coff)
V_max = vel_scale * np.abs(hydro_coff)
V_min = -vel_scale * np.abs(hydro_coff)

par_pos = []
par_vel = []
par_pbest = []
par_best_fitness = []

for i in range(N):
    x_0 = np.random.rand(len(hydro_coff)) * (2 * k * np.abs(hydro_coff)) + (
        hydro_coff - k * np.abs(hydro_coff)
    )  # 按照水动力系数允许的估计范围随机取粒子初始位置
    x_0 = np.random.rand(len(hydro_coff)) * (X_max - X_min) + X_min
    v_0 = np.random.rand(len(hydro_coff)) * (V_max - V_min) + V_min
    fitness_0 = 100000
    par_pos.append(x_0)  # 记录每个粒子的位置
    par_vel.append(v_0)  # 记录每个粒子的速度
    par_pbest.append(x_0)
    par_best_fitness.append(fitness_0)

gbest_fitness = par_best_fitness[np.argmin(np.abs(np.array(par_best_fitness)))]
gbest_fitness_list = [gbest_fitness]


c_1 = 2
c_2 = 3
epoch = 10000

for i in range(epoch):
    start = time.time()
    for j in range(N):

        par_pos_j = par_pos[j]
        par_vel_j = par_vel[j]
        bluerov.update_params(par_pos_j)
        vel_pred = bluerov.dynamics(motor)
        current_fitness = fitness_function(vel_pred, vel_true)

        if np.abs(current_fitness) < np.abs(
            par_best_fitness[j]
        ):  # 如果到了目前最好的适应度位置
            par_best_fitness[j] = current_fitness
            par_pbest[j] = par_pos_j  # 更新目前个体最好的位置
        minind = np.argmin(np.abs(np.array(par_best_fitness)))
        gbest_fitness = par_best_fitness[minind]  # 更新全局最优适合度
        gbest_pos = par_pbest[minind]  # 更新全局最优位置

        # if gbest_fitness != gbest_fitness_list[-1]:
        #     print(gbest_pos)

        gbest_fitness_list.append(gbest_fitness)
        delta_v = c_1 * np.random.rand(len(hydro_coff)) * (
            par_pbest[j] - par_pos_j
        ) + c_2 * np.random.rand(len(hydro_coff)) * (gbest_pos - par_pos_j)
        delta_v = np.clip(
            delta_v, -vel_scale * np.abs(par_pos_j), vel_scale * np.abs(par_pos_j)
        )

        par_vel_j = vel_weight * par_vel_j + delta_v
        par_pos_j += par_vel_j
        par_pos_j = np.clip(
            par_pos_j,
            hydro_coff - k * np.abs(hydro_coff),
            hydro_coff + k * np.abs(hydro_coff),
        )

        par_pos[j] = par_pos_j
        par_vel[j] = par_vel_j

    print(
        f"epoch {i+1}: the global best fitness is {gbest_fitness}, time:{time.time()-start}"
    )

gbest_fitness_list = np.array(gbest_fitness_list)

print(f"the best global hydro coff: {gbest_pos}")

plt.figure(1)
plt.plot(gbest_fitness_list, label="global best fitness")

bluerov.update_params(gbest_pos)
vel_pred = bluerov.dynamics(motor)
plot(vel_pred, vel_true)
