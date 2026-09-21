import numpy as np
from sim.common.quaternion import quat_derivative, quat_normalize, quat_multiply, quat_from_axis_angle


def accel_to_quat(accel_meas):
    ax, ay, az = -accel_meas / np.linalg.norm(accel_meas)
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    
    q_roll = quat_from_axis_angle(np.array([1,0,0]), roll)
    q_pitch = quat_from_axis_angle(np.array([0,1,0]), pitch)

    return quat_multiply(q_pitch, q_roll)

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro_meas, accel_meas, dt):
        q_dot = quat_derivative(self.q, gyro_meas)
        q_gyro = self.q + q_dot * dt
        q_gyro = quat_normalize(q_gyro)

        yaw = np.arctan2(2 * (q_gyro[0] * q_gyro[3] + q_gyro[1] * q_gyro[2]), 1 - 2 * (q_gyro[2] ** 2 + q_gyro[3] ** 2))
        q_yaw = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
        q_accel = quat_multiply(q_yaw, accel_to_quat(accel_meas))

        if np.dot(q_gyro, q_accel) < 0.0:
            q_accel = -q_accel

        q_blended = self.alpha * q_gyro + (1 - self.alpha) * q_accel
        self.q = quat_normalize(q_blended)

        return self.q

if __name__ == "__main__":
    from sim.plant.rigid_body import RigidBodyPlant
    from sim.sensors.imu_model import IMUModel

    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)
    imu = IMUModel()
    cf = ComplementaryFilter(alpha=0.98)

    torque = np.array([0.0, 0.0, 0.5])
    dt = 0.001

    for i in range(1000):
        plant.step(torque, dt)
        accel, gyro, mag = imu.sample(plant.state, dt)
        q_est = cf.update(gyro, accel, dt)

    print("true q:", plant.state[0:4])
    print("estimate q:", q_est)
