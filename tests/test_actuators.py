import numpy as np
from sim.actuators.motor_model import MotorModel
from sim.actuators.servo_model import ServoModel
from sim.actuators.geometry import total_torque


def test_motor_reaches_63_percent_after_one_tau():
    tau = 0.03
    motor = MotorModel(tau=tau, max_thrust=10.0)
    command = 5.0
    dt = 0.001

    for i in range(int(tau / dt)):
        motor.step(command, dt)

    assert np.isclose(motor.thrust, command * 0.63, atol=0.1)


def test_motor_saturates_at_max_thrust():
    motor = MotorModel(tau=0.03, max_thrust=10.0)
    dt = 0.001

    for i in range(1000):
        motor.step(50.0, dt)

    assert motor.thrust <= 10.0


def test_servo_respects_slew_rate():
    servo = ServoModel(slew_rate_deg=600.0)
    dt = 0.001

    for i in range(10):
        servo.step(np.radians(30.0), dt)

    assert np.isclose(np.degrees(servo.angle), 6.0)


def test_servo_settles_at_command():
    servo = ServoModel(slew_rate_deg=600.0)
    dt = 0.001

    for i in range(200):
        servo.step(np.radians(20.0), dt)

    assert np.isclose(np.degrees(servo.angle), 20.0)


def test_thrust_difference_produces_pure_roll():
    tau = total_torque(5.0, 0.0, 3.0, 0.0, arm_length=0.15, height=0.10)

    assert np.isclose(tau[0], 0.3)
    assert np.isclose(tau[1], 0.0)
    assert np.isclose(tau[2], 0.0)


def test_servo_tilt_produces_pure_pitch():
    tau = total_torque(4.0, np.radians(10), 4.0, np.radians(10),
                       arm_length=0.15, height=0.10)

    assert np.isclose(tau[0], 0.0)
    assert tau[1] < 0
    assert np.isclose(tau[2], 0.0)


def test_balanced_actuators_produce_no_torque():
    tau = total_torque(4.0, 0.0, 4.0, 0.0, arm_length=0.15, height=0.10)

    assert np.allclose(tau, np.zeros(3))
