import numpy as np
from sim.sensors.imu_model import IMUModel


def test_accel_measures_gravity_when_static():
    imu = IMUModel()
    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel, gyro, mag = imu.sample(identity_state, dt=0.001)

    assert np.isclose(accel[2], -9.81, atol=1.0)


def test_sample_is_random():
    imu = IMUModel()
    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel1, gyro1, mag1 = imu.sample(identity_state, dt=0.001)
    accel2, gyro2, mag2 = imu.sample(identity_state, dt=0.001)
    assert not np.allclose(accel1, accel2)


def test_gyro_bias_drifts_over_time():
    imu = IMUModel()
    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    inital_bias = imu.gyro_bias.copy()

    for i in range(1000):
        imu.sample(identity_state, dt=0.001)

    final_bias = imu.gyro_bias

    assert not np.allclose(inital_bias, final_bias)


def test_mag_magnitude_matches_field_strength():
    imu = IMUModel()
    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel, gyro, mag = imu.sample(identity_state, dt=0.001)

    assert np.isclose(np.linalg.norm(mag), 48000.0, atol=2000.0)