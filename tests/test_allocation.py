import numpy as np
from sim.control.allocation import (
    build_allocation_matrix,
    pseudo_inverse,
    allocate,
    saturate,
)
from sim.control.pid import PIDController
from sim.actuators.geometry import total_torque


L = 0.15
H = 0.10
T0 = 4.0


def _bp():
    B = build_allocation_matrix(L, H, T0)
    return B, pseudo_inverse(B)


def test_matrix_is_three_axis():
    B, _ = _bp()
    assert B.shape == (3, 4)
    assert np.allclose(B[0], [L, -L, 0, 0])
    assert np.allclose(B[1], [0, 0, -H * T0, -H * T0])
    assert np.allclose(B[2], [0, 0, L * T0, -L * T0])


def test_matrix_is_invertible():
    B, _ = _bp()
    assert not np.isclose(np.linalg.det(B @ B.T), 0.0)


def test_pseudo_inverse_recovers_desired_torque():
    B, Bp = _bp()
    tau_desired = np.array([0.2, -0.15, 0.08])
    assert np.allclose(B @ allocate(Bp, tau_desired), tau_desired)


def test_pure_roll_uses_only_thrust():
    _, Bp = _bp()
    u = allocate(Bp, np.array([0.3, 0.0, 0.0]))

    assert not np.isclose(u[0], 0.0)
    assert np.isclose(u[2], 0.0)
    assert np.isclose(u[3], 0.0)


def test_pure_pitch_uses_servos_in_same_direction():
    _, Bp = _bp()
    u = allocate(Bp, np.array([0.0, -0.1, 0.0]))

    assert np.isclose(u[0], 0.0)
    assert np.isclose(u[1], 0.0)
    assert np.isclose(u[2], u[3])
    assert not np.isclose(u[2], 0.0)


def test_pure_yaw_uses_servos_in_opposite_directions():
    _, Bp = _bp()
    u = allocate(Bp, np.array([0.0, 0.0, 0.1]))

    assert np.isclose(u[0], 0.0)
    assert np.isclose(u[1], 0.0)
    assert np.isclose(u[2], -u[3])
    assert not np.isclose(u[2], 0.0)


def test_allocation_matches_real_geometry():
    _, Bp = _bp()

    for tau in ([0.3, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.1]):
        u = allocate(Bp, np.array(tau, dtype=float))
        real = total_torque(T0 + u[0], u[2], T0 + u[1], u[3], L, H)
        assert np.allclose(real, tau, atol=1e-3)


def test_two_axis_matrix_still_available():
    B2 = build_allocation_matrix(L, H, T0, include_yaw=False)
    assert B2.shape == (2, 4)
    assert np.allclose(B2[0], [L, -L, 0, 0])


def test_saturation_clips_to_limits():
    _, Bp = _bp()
    max_dT = 0.5
    max_servo = np.radians(10.0)

    u = allocate(Bp, np.array([1.0, -0.5, 0.3]))
    u_sat = saturate(u, max_dT, max_servo)

    assert np.all(np.abs(u_sat[0:2]) <= max_dT + 1e-9)
    assert np.all(np.abs(u_sat[2:4]) <= max_servo + 1e-9)


def test_saturation_reduces_delivered_torque():
    B, Bp = _bp()
    tau_desired = np.array([1.0, -0.5, 0.3])
    u_sat = saturate(allocate(Bp, tau_desired), 0.5, np.radians(10.0))

    assert abs((B @ u_sat)[0]) < abs(tau_desired[0])


def test_antiwindup_stops_integral_growth():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limit=0.01)
    dt = 0.001

    for i in range(1000):
        pid.step(setpoint=1.0, measurement=0.0, dt=dt)

    assert pid.integral <= 0.011
