import numpy as np
from sim.sim_loop import SimLoop


def run(roll_sp_deg, pitch_sp_deg, yaw_sp_deg=0.0, steps=8000,
        disturbance_at=None, disturbance_axis=0, seed=0, **kw):
    sim = SimLoop(rng=np.random.default_rng(seed), **kw)
    sps = np.radians([roll_sp_deg, pitch_sp_deg, yaw_sp_deg])

    result = None
    for i in range(steps):
        t = i * sim.dt
        disturbance = None
        if disturbance_at is not None and disturbance_at <= t < disturbance_at + 0.05:
            disturbance = np.zeros(3)
            disturbance[disturbance_axis] = 0.3
        result = sim.step(*sps, disturbance_torque=disturbance)

    return sim, result


def _deg(result, key):
    return np.degrees(result[key])


def test_roll_reaches_setpoint():
    _, r = run(5.0, 0.0)
    assert np.isclose(_deg(r, "roll_true"), 5.0, atol=0.5)


def test_pitch_reaches_setpoint():
    _, r = run(0.0, 3.0)
    assert np.isclose(_deg(r, "pitch_true"), 3.0, atol=0.5)


def test_yaw_reaches_setpoint():
    _, r = run(0.0, 0.0, 10.0)
    assert np.isclose(_deg(r, "yaw_true"), 10.0, atol=0.5)


def test_all_three_axes_together():
    _, r = run(5.0, 3.0, 10.0)
    assert np.isclose(_deg(r, "roll_true"), 5.0, atol=0.6)
    assert np.isclose(_deg(r, "pitch_true"), 3.0, atol=0.6)
    assert np.isclose(_deg(r, "yaw_true"), 10.0, atol=0.6)


def test_axes_do_not_bleed_into_each_other():
    _, r = run(5.0, 0.0, 0.0)
    assert abs(_deg(r, "pitch_true")) < 0.5
    assert abs(_deg(r, "yaw_true")) < 0.5


def test_ekf_tracks_true_attitude():
    _, r = run(5.0, 3.0, 10.0)
    assert abs(_deg(r, "roll_true") - _deg(r, "roll_est")) < 0.5
    assert abs(_deg(r, "pitch_true") - _deg(r, "pitch_est")) < 0.5
    assert abs(_deg(r, "yaw_true") - _deg(r, "yaw_est")) < 0.5


def test_recovers_from_roll_disturbance():
    _, r = run(5.0, 0.0, steps=8000, disturbance_at=4.0, disturbance_axis=0)
    assert np.isclose(_deg(r, "roll_true"), 5.0, atol=0.5)


def test_recovers_from_yaw_disturbance():
    _, r = run(0.0, 0.0, steps=8000, disturbance_at=4.0, disturbance_axis=2)
    assert abs(_deg(r, "yaw_true")) < 0.5


def test_actuators_stay_within_limits():
    _, r = run(5.0, 3.0, 10.0)

    assert np.all(r["thrust"] >= 0.0)
    assert np.all(r["thrust"] <= 10.0)
    assert np.all(np.abs(r["servo"]) <= np.radians(20.0) + 1e-9)


def test_stabilises_inverted_pendulum():
    sim, r = run(0.0, 0.0, 0.0, steps=8000, initial_roll=np.radians(10.0))
    assert abs(_deg(r, "roll_true")) < 0.5


def test_diverges_without_controller():
    sim = SimLoop(rng=np.random.default_rng(0), initial_roll=np.radians(10.0))
    for pid in (sim.pid_roll, sim.pid_pitch, sim.pid_yaw):
        pid.kp = pid.ki = pid.kd = 0.0

    for _ in range(1200):
        sim.step(0.0, 0.0, 0.0)

    qw = abs(sim.plant.state[0])
    tilt = np.degrees(2 * np.arccos(np.clip(qw, 0.0, 1.0)))
    assert tilt > 90.0


def test_torque_free_mode_still_available():
    _, r = run(5.0, 0.0, 0.0, mass=0.0)
    assert np.isclose(_deg(r, "roll_true"), 5.0, atol=0.5)
