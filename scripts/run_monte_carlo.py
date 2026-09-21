import numpy as np

from sim.sim_loop import SimLoop


def run_single(gyro_std, accel_std, mag_std, init_roll, init_pitch,
               init_yaw=0.0, steps=6000, tolerance_deg=1.0, rng=None):
    sim = SimLoop(gyro_noise_std=gyro_std,
                  accel_noise_std=accel_std,
                  mag_noise_std=mag_std,
                  initial_roll=init_roll,
                  initial_pitch=init_pitch,
                  initial_yaw=init_yaw,
                  rng=rng)

    roll_sp = np.radians(5.0)
    pitch_sp = np.radians(0.0)
    yaw_sp = np.radians(0.0)

    result = None
    for i in range(steps):
        result = sim.step(roll_sp, pitch_sp, yaw_sp)

    roll_err = abs(np.degrees(result["roll_true"]) - 5.0)
    pitch_err = abs(np.degrees(result["pitch_true"]))
    yaw_err = abs(np.degrees(result["yaw_true"]))

    success = (roll_err < tolerance_deg and pitch_err < tolerance_deg
               and yaw_err < tolerance_deg)

    return {
        "success": success,
        "roll_err": roll_err,
        "pitch_err": pitch_err,
        "yaw_err": yaw_err,
    }


def monte_carlo(n_runs=100, seed=42):
    rng = np.random.default_rng(seed)

    successes = 0
    roll_errors = []
    pitch_errors = []
    yaw_errors = []

    for i in range(n_runs):
        gyro_std = rng.uniform(0.005, 0.03)
        accel_std = rng.uniform(0.02, 0.15)
        mag_std = rng.uniform(150.0, 600.0)
        init_roll = np.radians(rng.uniform(-15, 15))
        init_pitch = np.radians(rng.uniform(-15, 15))
        init_yaw = np.radians(rng.uniform(-30, 30))

        r = run_single(gyro_std, accel_std, mag_std, init_roll, init_pitch,
                       init_yaw=init_yaw, rng=rng)

        if r["success"]:
            successes += 1
        roll_errors.append(r["roll_err"])
        pitch_errors.append(r["pitch_err"])
        yaw_errors.append(r["yaw_err"])

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_runs} done")

    roll_errors = np.array(roll_errors)
    pitch_errors = np.array(pitch_errors)
    yaw_errors = np.array(yaw_errors)

    print("\n=== Monte Carlo results ===")
    print(f"runs            : {n_runs}")
    print(f"success rate    : {successes}/{n_runs} ({100*successes/n_runs:.1f}%)")
    print(f"roll error  mean: {roll_errors.mean():.3f} deg")
    print(f"roll error  max : {roll_errors.max():.3f} deg")
    print(f"pitch error mean: {pitch_errors.mean():.3f} deg")
    print(f"pitch error max : {pitch_errors.max():.3f} deg")
    print(f"yaw error   mean: {yaw_errors.mean():.3f} deg")
    print(f"yaw error   max : {yaw_errors.max():.3f} deg")

def run_dropout(dropout_rate, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    sim = SimLoop(rng=rng)

    errors = []

    for i in range(steps):
        t = i * sim.dt
        roll_sp = np.radians(5.0 * np.sin(2 * np.pi * 0.3 * t))
        pitch_sp = 0.0

        dropout = rng.random() < dropout_rate
        result = sim.step(roll_sp, pitch_sp, sensor_dropout=dropout)

        if t > 1.0:
            err = abs(np.degrees(result["roll_true"] - result["roll_est"]))
            errors.append(err)
    if not errors:
        return float("nan"), float("nan")

    return np.mean(errors), np.max(errors)

def run_jitter(jitter_fraction, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    sim = SimLoop(rng=rng)

    nominal_dt = sim.dt
    errors = []

    t = 0.0
    for i in range(steps):
        dt = nominal_dt * (1.0 + rng.uniform(-jitter_fraction, jitter_fraction))

        roll_sp = np.radians(5.0 * np.sin(2 * np.pi * 0.3 * t))
        result = sim.step(roll_sp, 0.0, dt_actual=dt)

        t += dt
        if t > 1.0:
            err = abs(np.degrees(result["roll_true"] - result["roll_est"]))
            errors.append(err)

    if not errors:
        return float("nan"), float("nan")

    return np.mean(errors), np.max(errors)


def jitter_sweep():
    print("\n=== Timing jitter sweep ===")
    print("jitter%   mean EKF error   max EKF error")

    for frac in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
        mean_err, max_err = run_jitter(frac)
        print(f"  {frac*100:5.1f}      {mean_err:8.3f}      {max_err:8.3f}")

def dropout_sweep():
    print("\n=== Sensor dropout sweep ===")
    print("dropout%   mean EKF error   max EKF error")

    for rate in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]:
        mean_err, max_err = run_dropout(rate)
        print(f"  {rate*100:5.1f}      {mean_err:8.3f}      {max_err:8.3f}")
def run_stall(stall_ms, stall_period_s, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    sim = SimLoop(rng=rng)
    nominal_dt = sim.dt

    errors = []
    t = 0.0
    next_stall = stall_period_s

    for i in range(steps):
        dt = nominal_dt
        if t >= next_stall:
            dt = stall_ms / 1000.0
            next_stall += stall_period_s

        roll_sp = np.radians(5.0 * np.sin(2 * np.pi * 0.3 * t))
        result = sim.step(roll_sp, 0.0, dt_actual=dt)

        t += dt
        if t > 1.0:
            err = abs(np.degrees(result["roll_true"] - result["roll_est"]))
            errors.append(err)

    if not errors:
        return float("nan"), float("nan")

    return np.mean(errors), np.max(errors)


def stall_sweep():
    print("\n=== Loop stall sweep ===")
    print("stall     mean EKF error   max EKF error")

    for stall_ms in [1, 5, 10, 20, 50, 100]:
        mean_err, max_err = run_stall(stall_ms, stall_period_s=0.5)
        print(f"  {stall_ms:4d}ms      {mean_err:8.3f}      {max_err:8.3f}")

def run_disturbance(disturbance_magnitude, steps=6000, seed=0):
    rng = np.random.default_rng(seed)
    sim = SimLoop(rng=rng)

    max_error = 0.0
    recovered = False

    for i in range(steps):
        t = i * sim.dt

        disturbance = None
        if 2.0 <= t < 2.1:
            disturbance = np.array([disturbance_magnitude, 0.0, 0.0])

        result = sim.step(np.radians(0.0), 0.0, disturbance_torque=disturbance)

        if t > 2.0:
            err = abs(np.degrees(result["roll_true"]))
            max_error = max(max_error, err)

    final_error = abs(np.degrees(result["roll_true"]))
    recovered = final_error < 1.0

    return max_error, final_error, recovered


def disturbance_sweep():
    print("\n=== Disturbance torque sweep ===")
    print("torque    max dev     final err  recovered")

    for mag in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        max_err, final_err, ok = run_disturbance(mag)
        status = "yes" if ok else "NO"
        print(f"  {mag:4.1f}     {max_err:7.2f}    {final_err:7.2f}     {status}")
if __name__ == "__main__":
    disturbance_sweep()