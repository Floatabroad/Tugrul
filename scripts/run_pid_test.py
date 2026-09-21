import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

from sim.plant.rigid_body import RigidBodyPlant
from sim.control.pid import PIDController


def quat_to_pitch(q):
    return 2 * np.arcsin(np.clip(q[2], -1.0, 1.0))


def main():
    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)

    pid = PIDController(kp=2.0, ki=2.0, kd=3.0, output_limit=0.1)

    dt = 0.001
    steps = 5000
    setpoint = np.radians(10.0)

    times = []
    angles = []
    commands = []

    for i in range(steps):
        pitch = quat_to_pitch(plant.state[0:4])
        u = pid.step(setpoint, pitch, dt)

        torque = np.array([0.0, u, 0.0])
        plant.step(torque, dt)

        times.append(i * dt)
        angles.append(np.degrees(pitch))
        commands.append(u)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, angles, label="pitch")
    ax1.axhline(np.degrees(setpoint), color='r', linestyle='--', label="target")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("pitch (deg)")
    ax1.set_title("PID step response")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, commands)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("torque command (N.m)")
    ax2.set_title("Controller output")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
