import numpy as np


def edf_force(thrust, servo_angle):
    return np.array([
        thrust * np.sin(servo_angle),
        0.0,
        -thrust * np.cos(servo_angle)
    ])


def total_torque(thrust1, servo1, thrust2, servo2, arm_length, height):
    r1 = np.array([0.0, -arm_length, -height])
    r2 = np.array([0.0, +arm_length, -height])

    f1 = edf_force(thrust1, servo1)
    f2 = edf_force(thrust2, servo2)

    return np.cross(r1, f1) + np.cross(r2, f2)

if __name__ == "__main__":
    L = 0.15
    h = 0.10

    tau = total_torque(5.0, 0.0, 3.0, 0.0, L, h)
    print("thrust difference (expect roll):", tau)

    tau = total_torque(4.0, np.radians(10), 4.0, np.radians(10), L, h)
    print("servo tilt (expect pitch):", tau)

    tau = total_torque(4.0, 0.0, 4.0, 0.0, L, h)
    print("balanced (expect zero):", tau)
