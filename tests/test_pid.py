import numpy as np
from sim.control.pid import PIDController
from sim.plant.rigid_body import RigidBodyPlant


def test_proportional_only_output():
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
    output = pid.step(setpoint=1.0, measurement=0.0, dt=0.001)

    assert np.isclose(output, 2.0, atol=0.01)


def test_integral_accumulates():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
    dt = 0.001

    for i in range(100):
        output = pid.step(setpoint=1.0, measurement=0.0, dt=dt)

    assert np.isclose(output, 0.1, atol=0.01)


def test_output_limit_saturates():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
    output = pid.step(setpoint=10.0, measurement=0.0, dt=0.001)

    assert np.isclose(output, 1.0)


def test_reset_clears_state():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
    dt = 0.001

    for i in range(100):
        pid.step(setpoint=1.0, measurement=0.0, dt=dt)

    pid.reset()
    output = pid.step(setpoint=1.0, measurement=0.0, dt=dt)

    assert np.isclose(output, 0.001, atol=0.001)


def test_closed_loop_reaches_setpoint():
    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)
    pid = PIDController(kp=2.0, ki=0.1, kd=3.0, output_limit=1.0)

    dt = 0.001
    setpoint = np.radians(10.0)

    for i in range(5000):
        pitch = 2 * np.arcsin(np.clip(plant.state[2], -1.0, 1.0))
        u = pid.step(setpoint, pitch, dt)
        plant.step(np.array([0.0, u, 0.0]), dt)

    final_pitch = 2 * np.arcsin(np.clip(plant.state[2], -1.0, 1.0))

    assert np.isclose(np.degrees(final_pitch), 10.0, atol=1.0)
