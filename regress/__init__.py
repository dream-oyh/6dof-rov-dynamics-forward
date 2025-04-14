import numpy as np
from robot import ROV, MOTOR
def build_reg_equation(bluerov: ROV, motor:MOTOR):
    u,v,w,p,q,r = bluerov.read_vel()
    X,Y,Z,K,M,N = motor.read_tau()
    m, J_x, J_y, J_z = bluerov.read_m()
    u_dot, v_dot, w_dot, p_dot, q_dot, r_dot = bluerov.read_acc()
    params_num = len(bluerov.hydro_coff)

    F = np.zeros((6 * len(bluerov.timestamps), 1))
    A = np.zeros((6 * len(bluerov.timestamps), params_num))

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
        
    return F, A


def objective_function(theta_prime, A_scaled, F_stack, lambda_reg):
    """
    岭回归的目标函数 (带约束优化用)。

    Args:
        theta_prime (np.ndarray): 当前迭代的缩放后参数向量 (M x 1)。
        A_scaled (np.ndarray): 缩放后的回归矩阵 (N_total x M)。
        F_stack (np.ndarray): F 向量 (N_total x 1)。
        lambda_reg (float): L2 正则化系数。

    Returns:
        float: 目标函数值。
    """
    residual = A_scaled @ theta_prime - F_stack.flatten() # Ensure F_stack is 1D
    lsq_term = 0.5 * np.sum(residual**2)
    reg_term = 0.5 * lambda_reg * np.sum(theta_prime**2)
    return lsq_term + reg_term

def objective_gradient(theta_prime, A_scaled, F_stack, lambda_reg):
    """
    岭回归目标函数的梯度 (雅可比矩阵)。

    Args:
        theta_prime (np.ndarray): 当前迭代的缩放后参数向量 (M x 1)。
        A_scaled (np.ndarray): 缩放后的回归矩阵 (N_total x M)。
        F_stack (np.ndarray): F 向量 (N_total x 1)。
        lambda_reg (float): L2 正则化系数。

    Returns:
        np.ndarray: 目标函数梯度 (M x 1)。
    """
    residual = A_scaled @ theta_prime - F_stack.flatten() # Ensure F_stack is 1D
    lsq_grad = A_scaled.T @ residual
    reg_grad = lambda_reg * theta_prime
    return lsq_grad + reg_grad