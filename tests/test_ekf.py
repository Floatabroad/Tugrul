import numpy as np
from sim.plant.rigid_body import RigidBodyPlant
from sim.sensors.imu_model import IMUModel
from sim.estimation.ekf import EKF


def run_sim(torque, steps=1000, dt=0.001):
    I = np.eye(3)
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)
    imu = IMUModel()
    ekf = EKF()

    for i in range(steps):
        plant.step(torque, dt)
        accel, gyro, mag = imu.sample(plant.state, dt)
        ekf.predict(gyro, dt)
        ekf.update(accel, mag)

    return plant, ekf


def test_ekf_tracks_roll():
    plant, ekf = run_sim(np.array([0.5, 0.0, 0.0]))

    true_qx = plant.state[1]
    est_qx = ekf.x[1]

    assert np.isclose(est_qx, true_qx, atol=0.02)


def test_ekf_yaw_does_not_drift():
    plant, ekf = run_sim(np.array([0.5, 0.0, 0.0]))

    est_qz = ekf.x[3]

    assert abs(est_qz) < 0.02

def test_quaternion_stays_normalized():
    plant, ekf = run_sim(np.array([0.5, 0.0, 0.0]))

    q = ekf.x[0:4]
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_update_reduces_covariance():
    ekf = EKF()
    imu = IMUModel()

    identity_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel, gyro, mag = imu.sample(identity_state, dt=0.001)

    ekf.predict(gyro, dt=0.001)
    P_after_predict = np.trace(ekf.P)

    ekf.update(accel, mag)
    P_after_update = np.trace(ekf.P)

    assert P_after_update < P_after_predict