import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

from sim.plant.rigid_body import RigidBodyPlant
from sim.sensors.imu_model import IMUModel
from sim.estimation.ekf import EKF, quat_angle_error


def main():
    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)
    imu = IMUModel()
    ekf = EKF()

    dt = 0.001
    steps = 60000

    times = []
    errors = []
    bias_history = []
    true_bias_history = []

    for i in range(steps):
        torque = np.array([0.3 * np.sin(2 * np.pi * 0.5 * i * dt), 0.0, 0.0])
        plant.step(torque, dt)
        accel, gyro, mag = imu.sample(plant.state, dt)
        ekf.predict(gyro, dt)
        ekf.update(accel, mag)

        error_rad = quat_angle_error(plant.state[0:4], ekf.x[0:4])

        times.append(i * dt)
        errors.append(np.degrees(error_rad))
        bias_history.append(ekf.x[4:7].copy())
        true_bias_history.append(imu.gyro_bias.copy())

    bias_history = np.array(bias_history)
    true_bias_history = np.array(true_bias_history)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    ax1 = axes[0]

    ax1.plot(times, errors)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("angle error (deg)")
    ax1.set_title("EKF attitude error")
    ax1.grid(True)

    labels = ['x', 'y', 'z']
    for i in range(3):
        ax = axes[i + 1]
        ax.plot(times, bias_history[:, i], label=f"estimate {labels[i]}")
        ax.plot(times, true_bias_history[:, i], '--', label=f"true {labels[i]}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("bias (rad/s)")
        ax.set_title(f"gyro bias - {labels[i]} axis")
        ax.legend()
        ax.grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
