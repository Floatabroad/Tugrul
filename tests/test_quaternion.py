import numpy as np
from sim.common.quaternion import quat_multiply, quat_normalize, quat_conjugate, quat_inverse, quat_to_rotmat, quat_derivative, quat_from_axis_angle


def test_quat_multiply_identity():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = quat_multiply(identity, identity)
    assert np.allclose(result, identity)


def test_quat_normalize_unit_length():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    result = quat_normalize(q)
    assert np.isclose(np.linalg.norm(result), 1.0)

def test_zero_guard():
    q = np.array([0.0, 0.0, 0.0, 0.0])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = quat_normalize(q)
    assert np.allclose(result, identity)
    
def test_conjugate():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_t = np.array([1.0, -2.0, -3.0, -4.0])
    q_conj = quat_conjugate(q)
    assert np.allclose(q_t, q_conj)

def test_inverse():
    q = np.array([1.0, 1.0, 1.0, 1.0])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q_inv = quat_inverse(q)
    result = quat_multiply(q, q_inv)
    assert np.allclose(identity, result)

def test_to_rotmat_identity():
    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    result = quat_to_rotmat(identity_q)
    assert np.allclose(result, np.eye(3))


def test_derivative_x_axis_spin():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    expected = np.array([0.0, 0.5, 0.0, 0.0])
    omega = np.array([1.0, 0.0, 0.0])
    result = quat_derivative(identity, omega)
    assert np.allclose(result, expected)

def test_axis_angle_90deg_about_z():
    ax_90 = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
    expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert np.allclose(ax_90, expected)