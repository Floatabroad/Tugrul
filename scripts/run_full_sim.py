import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

from sim.sim_loop import SimLoop


def main():
    sim = SimLoop()

    dt = sim.dt
    steps = 8000

    roll_sp = np.radians(5.0)
    pitch_sp = np.radians(3.0)
    yaw_sp = np.radians(10.0)

    times = []
    roll_true = []
    roll_est = []
    pitch_true = []
    yaw_true = []
    yaw_est = []
    thrusts = []
    servos = []

    for i in range(steps):
        t = i * dt

        disturbance = None
        if 4.0 <= t < 4.05:
            disturbance = np.array([0.3, 0.0, 0.0])

        result = sim.step(roll_sp, pitch_sp, yaw_sp, disturbance_torque=disturbance)

        times.append(t)
        roll_true.append(np.degrees(result["roll_true"]))
        roll_est.append(np.degrees(result["roll_est"]))
        pitch_true.append(np.degrees(result["pitch_true"]))
        yaw_true.append(np.degrees(result["yaw_true"]))
        yaw_est.append(np.degrees(result["yaw_est"]))
        thrusts.append(result["thrust"].copy())
        servos.append(result["servo"].copy())

    thrusts = np.array(thrusts)
    servos = np.array(servos)

    fig, axes = plt.subplots(5, 1, figsize=(11, 14))

    axes[0].plot(times, roll_true, label="true")
    axes[0].plot(times, roll_est, '--', label="EKF estimate")
    axes[0].axhline(np.degrees(roll_sp), color='r', linestyle=':', label="target")
    axes[0].set_ylabel("roll (deg)")
    axes[0].set_title("Roll tracking (disturbance impulse at 4 s)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, pitch_true)
    axes[1].axhline(np.degrees(pitch_sp), color='r', linestyle=':', label="target")
    axes[1].set_ylabel("pitch (deg)")
    axes[1].set_title("Pitch tracking")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(times, yaw_true, label="true")
    axes[2].plot(times, yaw_est, '--', label="EKF estimate")
    axes[2].axhline(np.degrees(yaw_sp), color='r', linestyle=':', label="target")
    axes[2].set_ylabel("yaw (deg)")
    axes[2].set_title("Yaw tracking (differential servo)")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(times, thrusts[:, 0], label="motor 1")
    axes[3].plot(times, thrusts[:, 1], label="motor 2")
    axes[3].set_ylabel("thrust (N)")
    axes[3].set_title("Motor commands")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(times, np.degrees(servos[:, 0]), label="servo 1")
    axes[4].plot(times, np.degrees(servos[:, 1]), label="servo 2")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("servo (deg)")
    axes[4].set_title("Servo commands (mean = pitch, difference = yaw)")
    axes[4].legend()
    axes[4].grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()