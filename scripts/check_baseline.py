import sys

import numpy as np

from sim.common.quaternion import (
    quat_multiply, quat_normalize, quat_conjugate, quat_inverse,
    quat_to_rotmat, quat_derivative, quat_from_axis_angle,
)
from sim.common.inertia import (
    box_inertia, cylinder_inertia, parallel_axis_shift, combine_inertia,
)
from sim.plant.integrators import rk4_step
from sim.plant.rigid_body import RigidBodyPlant
from sim.sensors.imu_model import IMUModel
from sim.estimation.ekf import EKF
from sim.control.pid import PIDController
from sim.control.allocation import (
    build_allocation_matrix, pseudo_inverse, allocate, saturate,
)
from sim.actuators.motor_model import MotorModel
from sim.actuators.servo_model import ServoModel
from sim.actuators.geometry import total_torque
from sim.common.frames import quat_to_euler, quat_from_euler
from sim.sim_loop import SimLoop


PASS = "  ok "
FAIL = "DRIFT"

SEED = 0


def rng():
    return np.random.default_rng(SEED)


def cmp(label, actual, expected, tol):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ok = np.all(np.abs(actual - expected) <= tol)
    tag = PASS if ok else FAIL
    a = np.array2string(actual, precision=5, suppress_small=True)
    e = np.array2string(expected, precision=5, suppress_small=True)
    print(f"  [{tag}] {label:<38} {a:<30} expected {e}")
    return ok


def section(name):
    print(f"\n--- {name} ---")


def check_quaternion():
    section("Phase 0 - quaternion")
    ok = True

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    ok &= cmp("identity x identity",
              quat_multiply(identity, identity), identity, 1e-12)
    ok &= cmp("normalize([2,0,0,0])",
              quat_normalize(np.array([2.0, 0, 0, 0])), identity, 1e-12)
    ok &= cmp("zero guard",
              quat_normalize(np.zeros(4)), identity, 1e-12)
    ok &= cmp("conjugate([1,2,3,4])",
              quat_conjugate(np.array([1.0, 2, 3, 4])),
              [1.0, -2, -3, -4], 1e-12)
    q = np.array([1.0, 1.0, 1.0, 1.0])
    ok &= cmp("q x inverse(q)",
              quat_multiply(q, quat_inverse(q)), identity, 1e-12)
    ok &= cmp("rotmat(identity) diag",
              np.diag(quat_to_rotmat(identity)), [1, 1, 1], 1e-12)
    ok &= cmp("derivative(identity, [1,0,0])",
              quat_derivative(identity, np.array([1.0, 0, 0])),
              [0, 0.5, 0, 0], 1e-12)
    ok &= cmp("axis_angle(z, 90deg)",
              quat_from_axis_angle(np.array([0, 0, 1.0]), np.pi / 2),
              [0.70711, 0, 0, 0.70711], 1e-4)
    return ok


def check_inertia():
    section("Phase 0 - inertia")
    ok = True

    ok &= cmp("cube(m=6, 2x2x2) diag",
              np.diag(box_inertia(6.0, 2.0, 2.0, 2.0)), [4, 4, 4], 1e-12)
    ok &= cmp("cylinder(m=2, r=1, h=4, z) diag",
              np.diag(cylinder_inertia(2.0, 1.0, 4.0, "z")),
              [3.16667, 3.16667, 1.0], 1e-4)

    I = box_inertia(1.0, 1.0, 1.0, 1.0)
    shifted = parallel_axis_shift(I, 1.0, np.array([2.0, 0, 0]))
    ok &= cmp("parallel axis, offset x=2",
              np.diag(shifted), [0.16667, 4.16667, 4.16667], 1e-4)

    ok &= cmp("combine(2I, 3I) diag",
              np.diag(combine_inertia([np.eye(3) * 2, np.eye(3) * 3])),
              [5, 5, 5], 1e-12)
    return ok


def check_integrator():
    section("Phase 1 - RK4")
    ok = True

    def f_exp(x):
        return x

    x = 1.0
    dt = 0.01
    for _ in range(100):
        x = rk4_step(f_exp, x, dt)

    ok &= cmp("exp(1) after 100 steps", [x], [np.e], 1e-9)
    return ok


def check_plant():
    section("Phase 1 - rigid body")
    ok = True

    plant = RigidBodyPlant(np.eye(3),
                           np.array([1.0, 0, 0, 0, 0, 0, 0]))
    torque = np.array([0.0, 0.0, 1.0])
    for _ in range(100):
        plant.step(torque, 0.01)

    ok &= cmp("omega after tau=1, t=1s",
              plant.state[4:7], [0, 0, 1.0], 1e-6)
    ok &= cmp("quaternion norm",
              [np.linalg.norm(plant.state[0:4])], [1.0], 1e-9)

    I_stand = np.diag([0.02, 0.02, 0.03])
    m, d, theta = 1.0, 0.02, np.radians(20.0)
    q20 = quat_from_euler(theta, 0.0, 0.0)

    below = RigidBodyPlant(I_stand, np.concatenate([q20, np.zeros(3)]),
                           mass=m, com_offset=[0, 0, d])
    ok &= cmp("gravity torque = -m g d sin(theta)",
              below.gravity_torque(q20),
              [-m * 9.81 * d * np.sin(theta), 0, 0], 1e-9)
    ok &= cmp("gravity torque zero when upright",
              below.gravity_torque(np.array([1.0, 0, 0, 0])), [0, 0, 0], 1e-12)

    angles = []
    for _ in range(6000):
        below.step(np.zeros(3), 0.001)
        angles.append(np.degrees(quat_to_euler(below.state[0:4])[0]))
    angles = np.array(angles)
    ok &= cmp("pendulum swings to far side", [angles.min()], [-20.0], 0.2)
    ok &= cmp("pendulum conserves energy", [np.abs(angles).max()], [20.0], 0.1)

    above = RigidBodyPlant(I_stand,
                           np.concatenate([quat_from_euler(np.radians(5.0), 0, 0),
                                           np.zeros(3)]),
                           mass=m, com_offset=[0, 0, -d])
    for _ in range(800):
        above.step(np.zeros(3), 0.001)
    w = np.sqrt(m * 9.81 * d / I_stand[0, 0])
    ok &= cmp("inverted pendulum grows as cosh(w t)",
              [np.degrees(quat_to_euler(above.state[0:4])[0]) / 5.0],
              [np.cosh(w * 0.8)], 0.15)
    return ok


def check_sensors():
    section("Phase 2 - IMU")
    ok = True

    imu = IMUModel(rng=rng())
    static = np.array([1.0, 0, 0, 0, 0, 0, 0])

    accels = []
    mags = []
    for _ in range(500):
        a, g, m = imu.sample(static, 0.001)
        accels.append(a)
        mags.append(m)

    accels = np.array(accels)
    mags = np.array(mags)

    ok &= cmp("accel z mean (NED, static)",
              [accels[:, 2].mean()], [-9.81], 0.02)
    ok &= cmp("accel xy mean",
              accels[:, 0:2].mean(axis=0), [0, 0], 0.02)
    ok &= cmp("mag magnitude mean",
              [np.linalg.norm(mags, axis=1).mean()], [48000.0], 500.0)
    ok &= cmp("bias drifted from zero",
              [float(np.linalg.norm(imu.gyro_bias) > 1e-9)], [1.0], 0.0)
    return ok


def check_actuators():
    section("Phase 4 - actuators")
    ok = True

    motor = MotorModel(tau=0.03, max_thrust=10.0)
    for _ in range(30):
        motor.step(5.0, 0.001)
    ok &= cmp("motor after 1 tau (63%)", [motor.thrust], [3.16060], 1e-4)

    for _ in range(60):
        motor.step(5.0, 0.001)
    ok &= cmp("motor after 3 tau (95%)", [motor.thrust], [4.75107], 1e-4)

    servo = ServoModel(slew_rate_deg=600.0)
    for _ in range(10):
        servo.step(np.radians(30.0), 0.001)
    ok &= cmp("servo after 10ms (slew)",
              [np.degrees(servo.angle)], [6.0], 1e-6)

    for _ in range(100):
        servo.step(np.radians(30.0), 0.001)
    ok &= cmp("servo settled", [np.degrees(servo.angle)], [30.0], 1e-6)

    L, h = 0.15, 0.10
    ok &= cmp("thrust diff -> pure roll",
              total_torque(5.0, 0.0, 3.0, 0.0, L, h), [0.3, 0, 0], 1e-9)
    ok &= cmp("servo tilt -> pure pitch",
              total_torque(4.0, np.radians(10), 4.0, np.radians(10), L, h),
              [0, -0.13892, 0], 1e-4)
    ok &= cmp("balanced -> zero torque",
              total_torque(4.0, 0.0, 4.0, 0.0, L, h), [0, 0, 0], 1e-12)
    return ok


def check_allocation():
    section("Phase 6 - allocation")
    ok = True

    L, h, T0 = 0.15, 0.10, 4.0
    B = build_allocation_matrix(L, h, T0)
    Bp = pseudo_inverse(B)

    ok &= cmp("B row0 (roll)", B[0], [0.15, -0.15, 0, 0], 1e-12)
    ok &= cmp("B row1 (pitch)", B[1], [0, 0, -0.4, -0.4], 1e-12)
    ok &= cmp("B row2 (yaw)", B[2], [0, 0, 0.6, -0.6], 1e-12)
    ok &= cmp("det(B B^T) nonzero",
              [float(not np.isclose(np.linalg.det(B @ B.T), 0.0))], [1.0], 0.0)

    u = allocate(Bp, np.array([0.3, 0.0, 0.0]))
    ok &= cmp("pure roll -> u", u, [1.0, -1.0, 0, 0], 1e-9)

    u = allocate(Bp, np.array([0.0, -0.1, 0.0]))
    ok &= cmp("pure pitch -> u (servos same direction)",
              u, [0, 0, 0.125, 0.125], 1e-9)

    u = allocate(Bp, np.array([0.0, 0.0, 0.1]))
    ok &= cmp("pure yaw -> u (servos opposite)",
              u, [0, 0, 0.08333, -0.08333], 1e-4)

    u = allocate(Bp, np.array([0.2, -0.15, 0.08]))
    ok &= cmp("round trip B @ u", B @ u, [0.2, -0.15, 0.08], 1e-9)

    for tau in ([0.3, 0, 0], [0, -0.1, 0], [0, 0, 0.1]):
        u = allocate(Bp, np.array(tau, dtype=float))
        real = total_torque(T0 + u[0], u[2], T0 + u[1], u[3], L, h)
        ok &= cmp(f"real geometry {tau}", real, tau, 1e-3)

    u = saturate(allocate(Bp, np.array([1.0, -0.5, 0.0])),
                 0.5, np.radians(10.0))
    ok &= cmp("saturated delivery", B @ u, [0.15, -0.13963, 0.0], 1e-4)
    return ok


def check_pid():
    section("Phase 5 - PID")
    ok = True

    pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
    ok &= cmp("P only, e=1", [pid.step(1.0, 0.0, 0.001)], [2.0], 0.02)

    pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
    out = 0.0
    for _ in range(100):
        out = pid.step(1.0, 0.0, 0.001)
    ok &= cmp("I accumulation 0.1s", [out], [0.1], 0.01)

    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
    ok &= cmp("output limit", [pid.step(10.0, 0.0, 0.001)], [1.0], 1e-12)

    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, output_limit=0.01)
    for _ in range(1000):
        pid.step(1.0, 0.0, 0.001)
    ok &= cmp("anti-windup caps integral",
              [float(pid.integral <= 0.011)], [1.0], 0.0)
    return ok


def check_ekf():
    section("Phase 3 - EKF")
    ok = True

    plant = RigidBodyPlant(np.eye(3), np.array([1.0, 0, 0, 0, 0, 0, 0]))
    imu = IMUModel(rng=rng())
    ekf = EKF()

    torque = np.array([0.5, 0.0, 0.0])
    for _ in range(1000):
        plant.step(torque, 0.001)
        a, g, m = imu.sample(plant.state, 0.001)
        ekf.predict(g, 0.001)
        ekf.update(a, m)

    ok &= cmp("roll tracking (qx)",
              [ekf.x[1]], [plant.state[1]], 0.02)
    ok &= cmp("yaw stays near zero (qz)",
              [abs(ekf.x[3])], [0.0], 0.02)
    ok &= cmp("bias magnitude sane",
              [float(np.linalg.norm(ekf.x[4:7]) < 0.01)], [1.0], 0.0)
    ok &= cmp("quaternion norm",
              [np.linalg.norm(ekf.x[0:4])], [1.0], 1e-9)
    ok &= cmp("P symmetric",
              [float(np.allclose(ekf.P, ekf.P.T, atol=1e-12))], [1.0], 0.0)
    ok &= cmp("P positive diagonal",
              [float(np.all(np.diag(ekf.P) > 0))], [1.0], 0.0)
    return ok


def check_full_loop():
    section("Phase 7 - closed loop")
    ok = True

    sim = SimLoop(rng=rng())
    r = None
    for _ in range(8000):
        r = sim.step(np.radians(5.0), np.radians(3.0), np.radians(10.0))

    ok &= cmp("roll reaches 5 deg",
              [np.degrees(r["roll_true"])], [5.0], 0.5)
    ok &= cmp("pitch reaches 3 deg",
              [np.degrees(r["pitch_true"])], [3.0], 0.5)
    ok &= cmp("yaw reaches 10 deg",
              [np.degrees(r["yaw_true"])], [10.0], 0.5)
    ok &= cmp("EKF tracks roll",
              [np.degrees(r["roll_true"] - r["roll_est"])], [0.0], 0.5)
    ok &= cmp("EKF tracks yaw",
              [np.degrees(r["yaw_true"] - r["yaw_est"])], [0.0], 0.5)
    ok &= cmp("thrust pair sums to 2*nominal",
              [r["thrust"].sum()], [8.0], 1e-6)
    ok &= cmp("thrust within actuator band",
              [float(np.all(np.abs(r["thrust"] - 4.0) <= 2.0 + 1e-9))],
              [1.0], 0.0)
    ok &= cmp("servos within limits",
              [float(np.all(np.abs(r["servo"]) <= np.radians(20) + 1e-9))],
              [1.0], 0.0)

    sim = SimLoop(rng=rng())
    for i in range(8000):
        t = i * sim.dt
        d = np.array([0.3, 0, 0]) if 4.0 <= t < 4.05 else None
        r = sim.step(np.radians(5.0), 0.0, 0.0, disturbance_torque=d)
    ok &= cmp("recovers after disturbance",
              [np.degrees(r["roll_true"])], [5.0], 0.5)

    sim = SimLoop(rng=rng())
    for i in range(8000):
        t = i * sim.dt
        d = np.array([0, 0, 0.2]) if 4.0 <= t < 4.05 else None
        r = sim.step(0.0, 0.0, 0.0, disturbance_torque=d)
    ok &= cmp("recovers after yaw disturbance",
              [np.degrees(r["yaw_true"])], [0.0], 0.5)
    return ok


def check_stress():
    section("Phase 8 - stress limits (short form)")
    ok = True

    for magnitude, expected, tol in ((0.5, 0.222, 0.05),
                                     (1.0, 0.963, 0.15),
                                     (2.0, 3.139, 0.40)):
        sim = SimLoop(rng=rng())
        r = None
        for i in range(4000):
            t = i * sim.dt
            d = np.array([magnitude, 0, 0]) if 2.0 <= t < 2.1 else None
            r = sim.step(0.0, 0.0, 0.0, disturbance_torque=d)
        ok &= cmp(f"{magnitude} Nm impulse -> residual error (deg)",
                  [abs(np.degrees(r["roll_true"]))], [expected], tol)
    return ok


def main():
    print("=" * 78)
    print("REGRESSION CHECK - comparison against reference values")
    print("=" * 78)

    results = {
        "quaternion": check_quaternion(),
        "inertia": check_inertia(),
        "integrator": check_integrator(),
        "plant": check_plant(),
        "sensors": check_sensors(),
        "actuators": check_actuators(),
        "allocation": check_allocation(),
        "pid": check_pid(),
        "ekf": check_ekf(),
        "full_loop": check_full_loop(),
        "stress": check_stress(),
    }

    print("\n" + "=" * 78)
    failed = [k for k, v in results.items() if not v]
    if failed:
        print("LAYERS THAT DRIFTED:", ", ".join(failed))
        print("\nStart from the topmost drifting layer - an error in a lower")
        print("layer propagates into the ones above it.")
    else:
        print("All layers match their reference values.")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
