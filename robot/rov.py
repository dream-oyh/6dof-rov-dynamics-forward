import numpy as np
import pandas as pd

from utils import Rzyx, S, Tzyx, update_pos

from .motor import MOTOR


class ROV:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pose_data = self._read_data()
        # self.timestamps = self.pose_data[:, 0]
        self.timestamps = np.array(range(len(self.pose_data))) * 0.05
        self.euler_angles = self.pose_data[:, 0:3]
        self.vel_b = np.concatenate(
            [self.pose_data[:, 3:6], self.pose_data[:, 6:9]], axis=1
        )
        self.acc = np.diff(self.vel_b, axis=0) / 0.05
        self.n_timestamps = len(self.timestamps)

        # self.mass = 11.2
        # self.x_size = 0.448
        # self.y_size = 0.2384
        # self.z_size = 0.28066

        # self.J_x = 0.2 * self.mass * self.y_size**2 + 0.2 * self.mass * self.z_size**2
        # self.J_y = 0.2 * self.mass * self.x_size**2 + 0.2 * self.mass * self.z_size**2
        # self.J_z = 0.2 * self.mass * self.x_size**2 + 0.2 * self.mass * self.y_size**2

        # self.X_udot = -1.7182
        # self.X_u = -11.7391
        # self.X_uu = 0

        # self.Y_vdot = 0
        # self.Y_v = -20
        # self.Y_vv = 0

        # self.Z_wdot = -5.468
        # self.Z_w = -31.8678
        # self.Z_ww = 0

        # self.K_pdot = 0
        # self.K_p = -25
        # self.K_pp = 0

        # self.M_qdot = -1.2481
        # self.M_q = -44.9085
        # self.M_qq = 0

        # self.N_rdot = -0.4006
        # self.N_r = -5
        # self.N_rr = 0

        self.mass = 11.5

        self.J_x = 0.16
        self.J_y = 0.16
        self.J_z = 0.16

        self.X_udot = -5.5
        self.X_u = -4.03
        self.X_uu = -18.18

        self.Y_vdot = -12.7
        self.Y_v = -6.22
        self.Y_vv = -21.66

        self.Z_wdot = -14.57
        self.Z_w = -5.18
        self.Z_ww = -36.99

        self.K_pdot = -0.12
        self.K_p = -0.07
        self.K_pp = -1.55

        self.M_qdot = -0.12
        self.M_q = -0.07
        self.M_qq = -1.55

        self.N_rdot = -0.12
        self.N_r = -0.07
        self.N_rr = -1.55

        # self.vel_b = self._transform_vel()
        self.r_bg = np.array([0, 0, 0.02], float)  # CG w.r.t. to the CO  重心向量
        # self.r_bb = np.array([0, 0, 0], float)  # CB w.r.t. to the CO  浮心向量

    def _read_data(self):
        pose_data = pd.read_csv(self.data_dir).to_numpy()
        return pose_data

    def _transform_vel(self):
        linear_velocity_inertial = np.zeros_like(self.vel_g[:, :3])
        angular_velocity_inertial = np.zeros_like(self.vel_g[:, 3:])

        for t in range(len(self.timestamps)):
            phi = self.euler_angles[t, 0]
            theta = self.euler_angles[t, 1]
            psi = self.euler_angles[t, 2]

            T = Tzyx(phi, theta)
            R = Rzyx(phi, theta, psi)

            linear_velocity_inertial[t, :] = np.linalg.inv(R) @ self.vel_g[t, :3]
            angular_velocity_inertial[t, :] = np.linalg.inv(T) @ self.vel_g[t, 3:]

        vel_b = np.concatenate(
            [linear_velocity_inertial, angular_velocity_inertial], axis=1
        )

        return vel_b

    @property
    def M_rb(self):
        M_rb = np.diag([self.mass, self.mass, self.mass, self.J_x, self.J_y, self.J_z])
        return M_rb

    @property
    def M_zg(self):
        return np.array(
            [
                [0, 0, 0, 0, self.mass * self.r_bg[2], 0],
                [0, 0, 0, -self.mass * self.r_bg[2], 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -self.mass * self.r_bg[2], 0, 0, 0, 0],
                [self.mass * self.r_bg[2], 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

    @property
    def M_a(self):
        return -np.diag(
            [
                self.X_udot,
                self.Y_vdot,
                self.Z_wdot,
                self.K_pdot,
                self.M_qdot,
                self.N_rdot,
            ]
        )

    def D(self, vel):
        D_1 = -np.diag(
            [
                self.X_u,
                self.Y_v,
                self.Z_w,
                self.K_p,
                self.M_q,
                self.N_r,
            ]
        )
        D_2 = -np.diag(
            [
                self.X_uu * np.abs(vel[0]),
                self.Y_vv * np.abs(vel[1]),
                self.Z_ww * np.abs(vel[2]),
                self.K_pp * np.abs(vel[3]),
                self.M_qq * np.abs(vel[4]),
                self.N_rr * np.abs(vel[5]),
            ]
        )
        return D_1 + D_2

    def calculate_Ca(self, mass, vel):

        v1 = vel[:3]
        v2 = vel[3:]
        M11 = mass[0:3, 0:3]
        M12 = mass[0:3, 3:6]
        M21 = M12.T
        M22 = mass[3:6, 3:6]
        C = np.zeros((6, 6))

        C[0:3, 3:] = -S(np.matmul(M11, v1) + np.matmul(M12, v2))
        C[3:, 0:3] = C[0:3, 3:]
        C[3:, 3:] = -S(np.matmul(M21, v1) + np.matmul(M22, v2))
        return C

    # def calculate_Crb(self, mass, vel):
    #     v1 = vel[:3]
    #     v2 = vel[3:]
    #     z_g = self.r_bg[2]
    #     m = mass[0, 0]
    #     J_x = mass[3, 3]
    #     J_y = mass[4, 4]
    #     J_z = mass[5, 5]
    #     C_rb = np.array(
    #         [
    #             [0, 0, 0, m * z_g * v2[2], m * v1[2], -m * v1[1]],
    #             [0, 0, 0, -m * v1[2], m * z_g * v2[2], m * v1[0]],
    #             [
    #                 0,
    #                 0,
    #                 0,
    #                 -m * (z_g * v2[0] - v1[1]),
    #                 -m * (z_g * v2[1] + v1[0]),
    #                 0,
    #             ],
    #             [
    #                 -m * z_g * v2[2],
    #                 m * v1[2],
    #                 m * (z_g * v2[0] - v1[1]),
    #                 0,
    #                 J_z * v2[2],
    #                 -J_y * v2[1],
    #             ],
    #             [
    #                 -m * v1[2],
    #                 -m * z_g * v2[2],
    #                 m * (z_g * v2[1] + v1[0]),
    #                 -J_z * v2[2],
    #                 0,
    #                 J_x * v2[0],
    #             ],
    #             [m * v1[1], -m * v1[0], 0, J_y * v2[1], -J_x * v2[0], 0],
    #         ]
    #     )
    #     return C_rb

    def calculate_buoyancy(self, euler):
        W = 112.8
        B = 114.8
        # W = self.mass * 9.81
        # B = 1028 * 9.81 * 0.011054
        phi = euler[0]
        theta = euler[1]
        sth = np.sin(theta)
        cth = np.cos(theta)
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        # g = np.array(
        #     [
        #         (W - B) * sth,
        #         -(W - B) * sphi * cth,
        #         -(W - B) * cphi * cth,
        #         0.01 * W * cth * sphi,
        #         0.01 * W * sth,
        #         0,
        #     ]
        # )
        g = np.array(
            [
                (W - B) * sth,
                -(W - B) * cth * sphi,
                -(W - B) * cth * cphi,
                self.r_bg[2] * W * cth * sphi,
                self.r_bg[2] * W * sth,
                0,
            ]
        )
        return g

    def dynamics(self, motor: MOTOR):
        v_pred = np.zeros((self.n_timestamps, 6))
        euler_pred = np.zeros((self.n_timestamps, 3))
        # set initial data
        v_pred[0, :] = self.vel_b[0, :]
        euler_pred[0, :] = self.euler_angles[0, :]
        M_rb = self.M_rb
        M_a = self.M_a
        M_zg = self.M_zg
        M = M_rb + M_a + M_zg

        for t in range(self.n_timestamps - 1):

            v_current = v_pred[t, :]
            euler_current = euler_pred[t, :3]

            C_rb = self.calculate_Ca(M_rb, v_current)
            C_a = self.calculate_Ca(M_a, v_current)
            g = self.calculate_buoyancy(euler_current)
            C_v = np.dot((C_rb + C_a), v_current)
            D_v = np.dot(self.D(v_current), v_current)

            tau = motor.tau[t, :]

            v_dot_real = (self.vel_b[t + 1, :] - self.vel_b[t, :]) / 0.05
            # tau = np.dot(M, v_dot_real) + C_v + D_v + g
            v_dot = np.linalg.inv(M) @ (tau - C_v - D_v - g)
            v_pred[t + 1, :] = v_current + v_dot * 0.05
            euler_pred[t + 1, :] = update_pos(euler_current, v_current, 0.05)

        return v_pred

    def update_params(self, params):
        self.X_udot = params[0]
        self.Y_vdot = params[1]
        self.Z_wdot = params[2]
        self.K_pdot = params[3]
        self.M_qdot = params[4]
        self.N_rdot = params[5]
        self.X_u = params[6]
        self.Y_v = params[7]
        self.Z_w = params[8]
        self.K_p = params[9]
        self.M_q = params[10]
        self.N_r = params[11]
        self.X_uu = params[12]
        self.Y_vv = params[13]
        self.Z_ww = params[14]
        self.K_pp = params[15]
        self.M_qq = params[16]
        self.N_rr = params[17]
