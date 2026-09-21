import numpy as np


def quat_to_euler(q):
    qw, qx, qy, qz = q

    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

    return roll, pitch, yaw


def quat_to_roll_pitch(q):
    roll, pitch, _ = quat_to_euler(q)
    return roll, pitch


def quat_to_yaw(q):
    return quat_to_euler(q)[2]


def quat_from_euler(roll, pitch, yaw=0.0):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_from_roll_pitch(roll, pitch):
    return quat_from_euler(roll, pitch, 0.0)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
