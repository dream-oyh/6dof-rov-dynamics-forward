import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MOTOR:
    def __init__(self, data_dir):
        # 转换前的 ENU 坐标系下的推力器方向
        r_f_enu = np.array(
            [
                [0.707, 0.707, 0],
                [0.707, -0.707, 0],
                [-0.707, 0.707, 0],
                [-0.707, -0.707, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]
        )

        # 转换前的 ENU 坐标系下的推力器位置
        r_p_enu = np.array(
            [
                [0.1355, -0.1, -0.0725],
                [0.1355, 0.1, -0.0725],
                [-0.1475, -0.1, -0.0725],
                [-0.1475, 0.1, -0.0725],
                [0.0025, -0.1105, -0.005],
                [0.0025, 0.1105, -0.005],
            ]
        )

        # 将推力器方向从 ENU 转换到 NED
        self.r_f = np.zeros_like(r_f_enu)
        for i in range(r_f_enu.shape[0]):
            self.r_f[i, 0] = r_f_enu[i, 1]  # y_enu -> x_ned
            self.r_f[i, 1] = r_f_enu[i, 0]  # x_enu -> y_ned
            self.r_f[i, 2] = -r_f_enu[i, 2]  # -z_enu -> z_ned

        # 将推力器位置从 ENU 转换到 NED
        self.r_p = np.zeros_like(r_p_enu)
        for i in range(r_p_enu.shape[0]):
            self.r_p[i, 0] = r_p_enu[i, 1]  # y_enu -> x_ned
            self.r_p[i, 1] = r_p_enu[i, 0]  # x_enu -> y_ned
            self.r_p[i, 2] = -r_p_enu[i, 2]  # -z_enu -> z_ned

        self.data_dir = data_dir
        self.tau = self._calculate_forces_and_torques()

    def _read_data(self):

        thrust_data = pd.read_csv(self.data_dir, sep=",").to_numpy()[:, 1:7]
        return thrust_data

    def _calculate_forces_and_torques(self):

        thrusts = self._read_data()
        n_timesteps = thrusts.shape[0]
        forces = np.zeros((n_timesteps, 3))
        torques = np.zeros((n_timesteps, 3))

        for t in range(n_timesteps):
            total_force = np.zeros(3)
            total_torque = np.zeros(3)

            for i in range(6):  # 遍历 6 个电机
                # 计算推力向量 (方向 * 大小)
                force_vector = self.r_f[i] * thrusts[t, i]
                # 累加合力
                total_force += force_vector
                # 计算力矩 (位置叉乘力)
                torque = np.cross(self.r_p[i], force_vector)
                total_torque += torque

            forces[t] = total_force
            torques[t] = total_torque

        return np.concatenate([forces, torques], axis=1)

    def visualize_coordinate_system(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 绘制原点
        ax.scatter(0, 0, 0, color="k", s=100)

        # 绘制坐标轴
        ax.quiver(0, 0, 0, 1, 0, 0, color="r", label="X轴")
        ax.quiver(0, 0, 0, 0, 1, 0, color="g", label="Y轴")
        ax.quiver(0, 0, 0, 0, 0, 1, color="b", label="Z轴")

        # 绘制推力器位置和方向
        for i in range(6):
            pos = self.r_p[i]
            dir = self.r_f[i]
            ax.scatter(pos[0], pos[1], pos[2], color="orange", s=50)
            ax.quiver(pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], color="purple")
            ax.text(pos[0], pos[1], pos[2], f"T{i+1}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("BlueROV2 坐标系和推力器配置")
        ax.legend()
        plt.show()
