import numpy as np

from sim.plant.rigid_body import RigidBodyPlant
from sim.sensors.imu_model import IMUModel
from sim.estimation.ekf import EKF
from sim.control.pid import PIDController
from sim.control.allocation import (
    build_allocation_matrix,
    pseudo_inverse,
    allocate,
    saturate,
)
from sim.actuators.motor_model import MotorModel
from sim.actuators.servo_model import ServoModel
from sim.actuators.geometry import total_torque
from sim.common.frames import (
    quat_to_euler, quat_to_roll_pitch, quat_from_euler, quat_from_roll_pitch,
    wrap_angle,
)

class SimLoop:

    def __init__(self, dt=0.001, arm_length=0.15, height=0.10, nominal_thrust=4.0,
                 gyro_noise_std=0.01, accel_noise_std=0.05, mag_noise_std=300.0,
                 initial_roll=0.0, initial_pitch=0.0, initial_yaw=0.0,
                 mass=1.0, com_offset=(0.0, 0.0, -0.02), rng=None):
        self.dt = dt
        self.arm_length = arm_length
        self.height = height
        self.nominal_thrust = nominal_thrust
        self.last_gyro = np.zeros(3)

        I = np.diag([0.02, 0.02, 0.03])

        q0 = quat_from_euler(initial_roll, initial_pitch, initial_yaw)
        initial_state = np.concatenate([q0, np.zeros(3)])
        self.plant = RigidBodyPlant(I, initial_state,
                                    mass=mass, com_offset=com_offset)

        self.imu = IMUModel(gyro_noise_std=gyro_noise_std,
                            accel_noise_std=accel_noise_std,
                            mag_noise_std=mag_noise_std,
                            rng=rng)
        self.ekf = EKF(gyro_noise_std=gyro_noise_std,
                       accel_noise_std=accel_noise_std,
                       mag_noise_std=mag_noise_std)

        self.pid_roll = PIDController(kp=4.0, ki=1.0, kd=1.0, output_limit=0.60,
                                      derivative_cutoff_hz=80.0)
        self.pid_pitch = PIDController(kp=1.2, ki=0.5, kd=0.2, output_limit=0.28,
                                       derivative_cutoff_hz=5.0)
        self.pid_yaw = PIDController(kp=0.6, ki=0.2, kd=0.2, output_limit=0.42,
                                     derivative_cutoff_hz=5.0)

        self.B = build_allocation_matrix(arm_length, height, nominal_thrust)
        self.B_pinv = pseudo_inverse(self.B)

        self.motor1 = MotorModel(tau=0.03, max_thrust=10.0)
        self.motor2 = MotorModel(tau=0.03, max_thrust=10.0)
        self.servo1 = ServoModel(slew_rate_deg=600.0, min_angle_deg=-20, max_angle_deg=20)
        self.servo2 = ServoModel(slew_rate_deg=600.0, min_angle_deg=-20, max_angle_deg=20)
        self.roll_saturated = False
        self.pitch_saturated = False
        self.yaw_saturated = False
        self.max_thrust_delta = 2.0
        self.max_servo_angle = np.radians(20.0)

    def step(self, roll_setpoint, pitch_setpoint, yaw_setpoint=0.0,
             disturbance_torque=None, sensor_dropout=False, dt_actual=None):
        dt_assumed = self.dt
        dt_real = self.dt if dt_actual is None else dt_actual

        accel, gyro, mag = self.imu.sample(self.plant.state, dt_real)

        if sensor_dropout:
            self.ekf.predict(self.last_gyro, dt_assumed)
        else:
            self.last_gyro = gyro
            self.ekf.predict(gyro, dt_assumed)
            self.ekf.update(accel, mag)

        roll_est, pitch_est, yaw_est = quat_to_euler(self.ekf.x[0:4])

        tau_roll = self.pid_roll.step(roll_setpoint, roll_est, dt_assumed,
                                      saturated=self.roll_saturated)
        tau_pitch = self.pid_pitch.step(pitch_setpoint, pitch_est, dt_assumed,
                                        saturated=self.pitch_saturated)
        yaw_error = wrap_angle(yaw_setpoint - yaw_est)
        tau_yaw = self.pid_yaw.step(0.0, -yaw_error, dt_assumed,
                                    saturated=self.yaw_saturated)

        tau_desired = np.array([tau_roll, tau_pitch, tau_yaw])

        u_raw = allocate(self.B_pinv, tau_desired)
        u = saturate(u_raw, self.max_thrust_delta, self.max_servo_angle)
        self.roll_saturated = not np.array_equal(u[0:2], u_raw[0:2])
        servo_saturated = not np.array_equal(u[2:4], u_raw[2:4])
        self.pitch_saturated = servo_saturated
        self.yaw_saturated = servo_saturated
        thrust1_cmd = self.nominal_thrust + u[0]
        thrust2_cmd = self.nominal_thrust + u[1]

        t1 = self.motor1.step(thrust1_cmd, dt_real)
        t2 = self.motor2.step(thrust2_cmd, dt_real)
        s1 = self.servo1.step(u[2], dt_real)
        s2 = self.servo2.step(u[3], dt_real)

        torque = total_torque(t1, s1, t2, s2, self.arm_length, self.height)

        if disturbance_torque is not None:
            torque = torque + disturbance_torque

        self.plant.step(torque, dt_real)

        roll_true, pitch_true, yaw_true = quat_to_euler(self.plant.state[0:4])

        return {
            "roll_true": roll_true,
            "pitch_true": pitch_true,
            "yaw_true": yaw_true,
            "roll_est": roll_est,
            "pitch_est": pitch_est,
            "yaw_est": yaw_est,
            "thrust": np.array([t1, t2]),
            "servo": np.array([s1, s2]),
            "torque": torque,
        }


if __name__ == "__main__":
    sim = SimLoop()

    roll_sp = np.radians(5.0)
    pitch_sp = np.radians(3.0)
    yaw_sp = np.radians(10.0)

    for i in range(8000):
        result = sim.step(roll_sp, pitch_sp, yaw_sp)

    for name, sp in (("roll", roll_sp), ("pitch", pitch_sp), ("yaw", yaw_sp)):
        print(f"{name:6s} target {np.degrees(sp):7.3f}  "
              f"true {np.degrees(result[name + '_true']):7.3f}  "
              f"est {np.degrees(result[name + '_est']):7.3f}")

    print()
    print("thrust      :", result["thrust"])
    print("servo (deg) :", np.degrees(result["servo"]))
