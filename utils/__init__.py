import math

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
