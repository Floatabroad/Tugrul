import numpy as np 
from sim.common.quaternion import quat_derivative, quat_normalize, quat_to_rotmat, quat_multiply, quat_inverse
from sim.sensors.imu_model import magnetic_field_ned

def vector_measurement_jacobian(q, v):
    qw, qx, qy, qz = q
    vx, vy, vz = v

    H = np.zeros((3, 7))

    H[0, 0:4] = 2 * np.array([
        vy*qz - vz*qy,
        vy*qy + vz*qz,
        -2*vx*qy + vy*qx - vz*qw,
        -2*vx*qz + vy*qw + vz*qx
    ])

    H[1, 0:4] = 2 * np.array([
        -vx*qz + vz*qx,
        vx*qy - 2*vy*qx + vz*qw,
        vx*qx + vz*qz,
        -vx*qw - 2*vy*qz + vz*qy
    ])

    H[2, 0:4] = 2 * np.array([
        vx*qy - vy*qx,
        vx*qz - vy*qw - 2*vz*qx,
        vx*qw + vy*qz - 2*vz*qy,
        vx*qx + vy*qy
    ])

    return H


def vector_measurement_model(q, v):
    R = quat_to_rotmat(q)
    return R.T @ v

def omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0]
    ])

def quat_omega_jacobian(q):
    qw, qx, qy, qz = q
    return np.array([
        [-qx, -qy, -qz],
        [qw, -qz, qy],
        [qz, qw, -qx],
        [-qy, qx, qw]
    ])


def quat_angle_error(q_true, q_est):
    q_diff = quat_multiply(q_true, quat_inverse(q_est))
    w = np.clip(abs(q_diff[0]), -1.0, 1.0)
    return 2 * np.arccos(w)

class EKF:
    def __init__(self, gyro_noise_std=0.01, gyro_bias_walk_std=0.0001,
                 accel_noise_std=0.05, mag_noise_std=300.0, gravity=9.81,
                 mag_declination_deg=6.0, mag_inclination_deg=58.0,
                 mag_strength=48000.0):
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(7) * 0.01
        self.P[4:7, 4:7] = np.eye(3) * 1e-5

        self.gyro_noise_var = gyro_noise_std**2
        self.gyro_bias_walk_var = gyro_bias_walk_std**2

        self.R = np.eye(6)
        self.R[0:3, 0:3] = np.eye(3) * accel_noise_std ** 2
        self.R[3:6, 3:6] = np.eye(3) * mag_noise_std ** 2

        self.gravity = gravity
        self.gravity_world = np.array([0.0, 0.0, -gravity])
        self.mag_world = magnetic_field_ned(mag_declination_deg,
                                            mag_inclination_deg,
                                            mag_strength)

    def predict(self, gyro_meas, dt):
        q = self.x[0:4]
        bias = self.x[4:7]

        omega = gyro_meas - bias

        q_dot = quat_derivative(q, omega)
        q_new = quat_normalize(q + q_dot * dt)

        self.x = np.concatenate([q_new, bias])

        F = np.eye(7)
        F[0:4, 0:4] = np.eye(4)+ 0.5 * dt * omega_matrix(omega)
        F[0:4, 4:7] = -0.5 * dt * quat_omega_jacobian(q)
        G = quat_omega_jacobian(q)

        Q = np.zeros((7, 7))
        Q[0:4, 0:4] = 0.25 * dt * self.gyro_noise_var * (G @ G.T) + np.eye(4) * 1e-12
        Q[4:7, 4:7] = self.gyro_bias_walk_var * dt * 100 *np.eye(3)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, accel_meas, mag_meas):
        q = self.x[0:4]

        h_accel = vector_measurement_model(q, self.gravity_world)
        h_mag = vector_measurement_model(q, self.mag_world)
        h = np.concatenate([h_accel, h_mag])

        H_accel = vector_measurement_jacobian(q, self.gravity_world)
        H_mag = vector_measurement_jacobian(q, self.mag_world)
        H = np.vstack([H_accel, H_mag])

        z = np.concatenate([accel_meas, mag_meas])
        y = z - h

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        q_new = quat_normalize(self.x[0:4])

        if np.dot(q_new, q) < 0:
            q_new = -q_new

        self.x = np.concatenate([q_new, self.x[4:7]])

        I7 = np.eye(7)
        self.P = (I7 - K @ H) @ self.P @ (I7 - K @ H).T + K @ self.R @ K.T


if __name__ == "__main__":
    from sim.plant.rigid_body import RigidBodyPlant
    from sim.sensors.imu_model import IMUModel

    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)
    imu = IMUModel()
    ekf = EKF()
    torque = np.array([0.5, 0.0, 0.0])
    dt = 0.001

    for i in range(1000):
        plant.step(torque, dt)
        accel, gyro, mag = imu.sample(plant.state, dt)
        ekf.predict(gyro, dt)
        ekf.update(accel, mag)

    print("true q:", plant.state[0:4])
    print("EKF q:   ", ekf.x[0:4])
    print("EKF bias:", ekf.x[4:7])
