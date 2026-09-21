import numpy as np
from sim.plant.rigid_body import RigidBodyPlant


def test_constant_torque_z_axis():
    I = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)

    torque = np.array([0.0, 0.0, 1.0])
    dt = 0.01
    steps = 100

    for i in range(steps):
        plant.step(torque, dt)

    total_time = dt * steps
    omega_z_analytic = torque[2] / I[2, 2] * total_time

    simulated_omega = plant.state[4:7]
    expected_omega = np.array([0.0, 0.0, omega_z_analytic])

    assert np.allclose(simulated_omega, expected_omega)


def test_quaternion_stays_normalized():
    I = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)

    torque = np.array([0.0, 0.0, 1.0])
    dt = 0.01
    steps = 100

    for i in range(steps):
        plant.step(torque, dt)

    q_final = plant.state[0:4]
    assert np.isclose(np.linalg.norm(q_final), 1.0)


I_STAND = np.diag([0.02, 0.02, 0.03])


def _at_roll(deg, **kw):
    from sim.common.frames import quat_from_roll_pitch
    q = quat_from_roll_pitch(np.radians(deg), 0.0)
    return RigidBodyPlant(I_STAND, np.concatenate([q, np.zeros(3)]), **kw)


def test_no_gravity_torque_by_default():
    plant = _at_roll(20.0)
    for _ in range(5000):
        plant.step(np.zeros(3), 0.001)

    from sim.common.frames import quat_to_roll_pitch
    assert np.isclose(np.degrees(quat_to_roll_pitch(plant.state[0:4])[0]), 20.0)


def test_gravity_torque_matches_analytic():
    from sim.common.frames import quat_from_roll_pitch
    m, d, theta = 1.0, 0.02, np.radians(20.0)

    plant = _at_roll(20.0, mass=m, com_offset=[0, 0, d])
    tau = plant.gravity_torque(quat_from_roll_pitch(theta, 0.0))

    assert np.isclose(tau[0], -m * 9.81 * d * np.sin(theta))
    assert np.allclose(tau[1:], 0.0)


def test_gravity_torque_zero_when_upright():
    plant = _at_roll(0.0, mass=1.0, com_offset=[0, 0, 0.02])
    assert np.allclose(plant.gravity_torque(np.array([1.0, 0, 0, 0])), 0.0)


def test_com_below_pivot_is_a_stable_pendulum():
    from sim.common.frames import quat_to_roll_pitch
    plant = _at_roll(20.0, mass=1.0, com_offset=[0, 0, 0.02])

    angles = []
    for _ in range(6000):
        plant.step(np.zeros(3), 0.001)
        angles.append(np.degrees(quat_to_roll_pitch(plant.state[0:4])[0]))

    angles = np.array(angles)
    assert angles.min() < -15.0
    assert np.abs(angles).max() <= 20.1


def test_com_above_pivot_is_an_unstable_inverted_pendulum():
    from sim.common.frames import quat_to_roll_pitch
    m, d = 1.0, 0.02
    plant = _at_roll(5.0, mass=m, com_offset=[0, 0, -d])

    for _ in range(800):
        plant.step(np.zeros(3), 0.001)

    grown = np.degrees(quat_to_roll_pitch(plant.state[0:4])[0]) / 5.0
    w = np.sqrt(m * 9.81 * d / I_STAND[0, 0])
    assert np.isclose(grown, np.cosh(w * 0.8), rtol=0.05)


def test_gravity_torque_does_not_act_on_yaw():
    from sim.common.frames import quat_from_euler
    plant = _at_roll(0.0, mass=1.0, com_offset=[0, 0, -0.02])

    for roll, pitch in ((0.3, 0.0), (0.0, 0.3), (0.2, -0.4)):
        tau = plant.gravity_torque(quat_from_euler(roll, pitch, 0.7))
        assert np.isclose(tau[2], 0.0)
