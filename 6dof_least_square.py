import numpy as np
from scipy.optimize import minimize

from plot import plot
from regress import build_reg_equation, objective_function, objective_gradient
from robot import MOTOR, ROV
from utils import output, std, z_score

pose_dir = "20250410_data/pose_data.csv"
thruster_dir = "20250410_data/thrust_data.csv"

bluerov = ROV(pose_dir)
motor = MOTOR(thruster_dir)
params_num = len(bluerov.hydro_coff)

F, A = build_reg_equation(bluerov, motor)
lambda_reg = 1400

A_std = std(A)
bounds = [(None, 0) for _ in range(params_num)]
theta = np.linalg.lstsq(z_score(A), F, rcond=None)[0]
theta = np.squeeze(theta) / A_std
theta_prime_initial_guess = np.random.rand(params_num) * 5

optimization_result = minimize(
    fun=objective_function,  # 目标函数
    x0=theta_prime_initial_guess,  # 初始猜测
    args=(A, F, lambda_reg),  # 传递给目标函数和梯度的额外参数
    method="SLSQP",  # 优化方法
    jac=objective_gradient,  # 梯度函数 (雅可比)
    bounds=bounds,  # 参数边界约束
    options={
        "disp": True,
        "maxiter": 1000,
        "ftol": 1e-8,
    },  # 显示优化过程，设置迭代次数和容忍度
)

if optimization_result.success:
    theta_prime_optimal = optimization_result.x
    print("\nOptimization successful!")
    print(f"Optimal scaled parameters (theta_prime): \n{theta_prime_optimal}")

    theta_optimal = theta_prime_optimal / A_std
    print(f"\nOptimal physical parameters (theta): \n{theta_optimal}")

    output("results/identify_results.md", theta, bluerov)

else:
    print("\nOptimization failed!")
    print(f"Message: {optimization_result.message}")

bluerov.update_params(theta_optimal)
vel_pred = bluerov.dynamics(motor)
plot(vel_pred, bluerov.vel_b)
