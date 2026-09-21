import numpy as np
from sim.common.quaternion import quat_to_rotmat

def magnetic_field_ned(declination_deg, inclination_deg, strength):
    dec = np.radians(declination_deg)
    inc = np.radians(inclination_deg)

    north = strength * np.cos(inc) * np.cos(dec)
    east  = strength * np.cos(inc) * np.sin(dec)
    down  = strength * np.sin(inc)

    return np.array([north, east, down])

class IMUModel:
    def __init__(self, gyro_noise_std=0.01, accel_noise_std=0.05,
                 gyro_bias_walk_std=0.0001, gravity=9.81,
                 mag_declination_deg=6.0, mag_inclination_deg=58.0,
                 mag_strength=48000.0, mag_noise_std=300.0, rng=None):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_walk_std = gyro_bias_walk_std
        self.gravity = gravity
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.rng = np.random.default_rng(rng)
        self.mag_noise_std = mag_noise_std
        self.mag_world = magnetic_field_ned(mag_declination_deg,
                                            mag_inclination_deg,
                                            mag_strength)

    def sample(self, true_state, dt):
        q = true_state[0:4]
        true_omega = true_state[4:7]

        bias_step = self.rng.normal(0.0, self.gyro_bias_walk_std * np.sqrt(dt), size=3)
        self.gyro_bias = self.gyro_bias + bias_step

        gyro_noise = self.rng.normal(0.0, self.gyro_noise_std, size=3)
        gyro_meas = true_omega + self.gyro_bias + gyro_noise

        R = quat_to_rotmat(q)
        gravity_world = np.array([0.0, 0.0, -self.gravity])
        gravity_body  = R.T @ gravity_world
        accel_noise = self.rng.normal(0.0, self.accel_noise_std, size=3)
        accel_meas = gravity_body + accel_noise

        mag_body = R.T @ self.mag_world
        mag_noise = self.rng.normal(0.0, self.mag_noise_std, size=3)
        mag_meas = mag_body + mag_noise

        return accel_meas, gyro_meas, mag_meas


if __name__ == "__main__":
    imu = IMUModel()

    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel, gyro, mag = imu.sample(identity_state, dt=0.001)
    print("accel:", accel)
    print("gyro:", gyro)
    print("mag:", mag)