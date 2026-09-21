import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2 

    return np.array([w,x,y,z])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    conj = quat_conjugate(q)
    norm_sq = np.linalg.norm(q) ** 2 
    return conj / norm_sq


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w,x,y,z, = q

    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)

    r10 = 2*(x*y+w*z)
    r11 = 1-2*(x**2+z**2)
    r12 = 2*(y*z-w*x)
    
    r20 = 2*(x*z-w*y)
    r21 = 2*(y*z+w*x)
    r22 = 1-2*(x**2+y**2)
    
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
        ])


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    wx, wy, wz = omega
    omega_quat = np.array([0.0, wx, wy, wz])
    q_dot= 0.5 * (quat_multiply(q, omega_quat))
    return q_dot


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_norm = axis / np.linalg.norm(axis)
    ax, ay, az = axis_norm

    half_angle = angle_rad / 2
    w = np.cos(half_angle)
    x = ax * np.sin(half_angle)
    y = ay * np.sin(half_angle)
    z = az * np.sin(half_angle)

    return np.array([w, x, y, z])


if __name__ == "__main__":
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    zero = np.array([0.0, 0.0, 0.0, 0.0])
    result = quat_multiply(identity, identity)
    zerotest = quat_normalize(zero)
    q_test = np.array([2.0, 0.0, 0.0, 0.0])
    normal_q = quat_normalize(q_test)
    conjugate_test = quat_multiply(normal_q, quat_conjugate(normal_q))
    inverse_test = quat_multiply(q_test, quat_inverse(q_test))
    conversion_test = quat_to_rotmat(identity) 
    dO_ex = np.array([1.0,0.0,0.0])
    d_Test = quat_derivative(identity, dO_ex)
    ax_test = np.array([0,0,1])
    ax_arr = quat_from_axis_angle(ax_test, 0)
    ax_90 = quat_from_axis_angle(np.array([0,0,1]), np.pi/2)
    print(ax_90)
    print(ax_arr)
    print(d_Test)
    print(conversion_test)    
    print(inverse_test)
    print(conjugate_test)
    print(normal_q)
    print(result)
    print(zerotest)
