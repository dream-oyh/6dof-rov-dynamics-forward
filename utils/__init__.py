import datetime
import math
import os

import numpy as np



def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + sth * spsi * cphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    return R


def Tzyx(phi, theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)

    try:
        T = np.array(
            [
                [1, sphi * sth / cth, cphi * sth / cth],
                [0, cphi, -sphi],
                [0, sphi / cth, cphi / cth],
            ]
        )

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return T


def S(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b.
    """

    S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return S


def H(r):
    """
    H = Hmtrx(r) computes the 6x6 system transformation matrix
    H = [eye(3)     S'
         zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

    If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
    CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
    force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG
    """

    H = np.identity(6, float)
    H[0:3, 3:6] = S(r).T

    return H


def update_pos(pos, v_b, sample_time):
    phi = pos[0]
    theta = pos[1]
    psi = pos[2]
    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta)
    v_n = np.zeros(6)
    v_n[:3] = R @ v_b[:3]
    v_n[3:] = T @ v_b[3:]
    pos += v_n[3:] * sample_time
    return pos


def ENU2NED(enu_data):
    ned_data = enu_data.copy()
    ned_data[:, 0] = enu_data[:, 1]
    ned_data[:, 1] = enu_data[:, 0]
    ned_data[:, 2] = -enu_data[:, 2]
    return ned_data


def output(output_dir, theta, bluerov):

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
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


def std(A):
    A_std = np.std(A, axis=0)
    A_std[A_std < 1e-8] = 1e-8
    return A_std


def mean(A):
    return np.mean(A, axis=0)


def z_score(A):
    return (A - mean(A)) / std(A)


def read_columns(matrix):
    num_cols = matrix.shape[1]
    return tuple(matrix[:, i] for i in range(num_cols))
