# Dual-EDF Gimbal Thrust Stand — Simulation

A flight-control chain for a jetpack-style attitude control system, written
from scratch in Python. No control or estimation libraries: quaternion algebra,
rigid-body dynamics, an EKF with hand-derived Jacobians, and control allocation
are all implemented directly.

The goal is to validate the whole chain before moving to hardware
(STM32F411 + MPU9250 + 2 EDF motors + 2 servos). The estimator and controller
are written so they port to C line by line; only the sensor and actuator layers
get replaced with real I/O.

## The system

A gimbal on a pivot, driven by two ducted fans on a common arm. Each fan sits on
a servo that tilts its thrust vector.

```
          x (forward)
          ^
          |
   o------+------o  --->  y (right)
 EDF1          EDF2
 (-L)          (+L)

   z points down, into the page (NED)
```

Four actuators, three controlled axes:

| Axis | Produced by | Torque |
|---|---|---|
| Roll | thrust difference | `L (T1 cosθ1 - T2 cosθ2)` |
| Pitch | both servos tilting **together** | `-h (T1 sinθ1 + T2 sinθ2)` |
| Yaw | both servos tilting **opposite** | `L (T1 sinθ1 - T2 sinθ2)` |

Pitch and yaw share the same two servos but use orthogonal directions of the
servo space — their mean and their difference — so the pseudo-inverse separates
them cleanly.

The plant is an **inverted pendulum**: the centre of mass sits above the pivot,
so gravity actively destabilises it. Left alone it falls away as
`θ₀·cosh(ωt)` with `ω = sqrt(m·g·d/I) ≈ 3.1 rad/s`. Holding it upright is the
actual control problem.

## Quick start

```bash
uv sync --all-groups
```

Then either use the menu:

```bash
uv run python app.py            # interactive menu
uv run python app.py --help     # list the modes
uv run python app.py sandbox    # jump straight to a mode
```

...or run the pieces directly:

```bash
uv run pytest -v                          # 65 tests
uv run python -m scripts.check_baseline   # deterministic regression check

uv run python -m scripts.live_sim         # live sim, sliders change it while running
uv run python -m scripts.sandbox          # 3D scene, right-drag to push the body
uv run python -m scripts.tuner            # compare two gain sets side by side
uv run python -m scripts.run_full_sim     # closed loop + plots
uv run python -m scripts.run_monte_carlo  # stress sweeps
```

Modules importing `sim.*` must be run with `-m`, not by file path.

A standalone binary needing no Python at all can be built with PyInstaller:

```bash
uv run pyinstaller --onefile --name gimbal-sim \
  --hidden-import matplotlib.backends.backend_tkagg \
  --hidden-import PIL._tkinter_finder \
  --collect-submodules sim --collect-submodules scripts \
  --noconfirm --clean app.py
```

## Layout

```
sim/
├── common/
│   ├── quaternion.py      Hamilton product, normalize, rotation matrix, derivative
│   ├── frames.py          quaternion <-> roll/pitch/yaw, angle wrapping
│   ├── inertia.py         box/cylinder inertia, parallel axis
│   └── config.py          sensor and hardware constants
├── plant/
│   ├── integrators.py     RK4
│   ├── rigid_body.py      Euler rotational dynamics + gravity torque
│   └── visualize.py       matplotlib 3D
├── sensors/
│   └── imu_model.py       noisy accel + gyro + magnetometer, bias random walk
├── estimation/
│   ├── ekf.py             7-state EKF, hand-derived Jacobians
│   └── complementary_filter.py   reference implementation, shows why the EKF is needed
├── control/
│   ├── pid.py             anti-windup + derivative filter
│   └── allocation.py      3x4 allocation matrix, pseudo-inverse, saturation
├── actuators/
│   ├── motor_model.py     first-order lag
│   ├── servo_model.py     slew-rate limit
│   └── geometry.py        thrust + angle -> torque
└── sim_loop.py            full closed loop

scripts/   runnable simulations, plots and interactive tools
tests/     pytest, 65 tests
```

## Estimator

**State (7):** `[q(4), gyro_bias(3)]`

**Predict** integrates the gyro and propagates the covariance through the `F`
Jacobian; process noise is derived through the quaternion rate Jacobian.

**Update** fuses a 6-dimensional measurement: the accelerometer gives the
gravity direction, the magnetometer gives magnetic north. Covariance uses the
Joseph form.

The magnetometer is not optional. With accelerometer and gyro alone the yaw axis
is unobservable — gravity is invariant under rotation about the vertical, so the
filter gets no correction there. Before the magnetometer was added, yaw drifted
by 0.19 rad and the z component of the estimated gyro bias came out twenty times
larger than the others, the filter mistaking noise for signal. Afterwards the
drift fell to 0.005 rad.

## Controller

| Axis | Actuator | kp | ki | kd | limit (N·m) | derivative cutoff |
|---|---|---|---|---|---|---|
| Roll | motor difference | 4.0 | 1.0 | 1.0 | 0.60 | 80 Hz |
| Pitch | servo mean | 1.2 | 0.5 | 0.2 | 0.28 | 5 Hz |
| Yaw | servo difference | 0.6 | 0.2 | 0.2 | 0.42 | 5 Hz |

Output limits equal the real actuator authority, so the PID's own clamp and
actuator saturation coincide and conditional-integration anti-windup engages at
the right point.

Gains differ per axis because the actuators differ: roll is driven by motors
(30 ms lag), pitch and yaw by slew-rate-limited servos. On an unstable plant the
proportional gain must also exceed the destabilising gravity gradient
`m·g·d = 0.196 N·m/rad`, otherwise steady-state error becomes a gain error
rather than an offset.

## Validation

All noise sources are seeded; the same seed reproduces the same numbers.
`scripts/check_baseline` re-derives every reference value and exits non-zero if
any layer has drifted.

| Test | Result |
|---|---|
| Monte Carlo, 100 runs, random noise and initial attitude | **100/100**, worst-case error 0.47° on any axis |
| Sensor dropout | 0.29° worst-case estimate error even at 90% loss |
| Random timing jitter | no measurable effect up to ±50% |
| Single control-loop stall | no effect to 20 ms, 4.3° at 100 ms |
| Disturbance torque rejection | recovers from 1.0 N·m (roll authority is 0.6 N·m) |
| Inverted pendulum | diverges open loop, held within 0.5° closed loop |

## Known limitations

- The allocation linearisation holds to about ±20° of servo deflection
- Pitch and yaw share the same servos; under saturation they compete, and there
  is no prioritised allocation
- Inertia tensor, mass and CoM offset are estimates and must be measured on the
  real stand
- EKF tuning follows a synthetic gyro model; real hardware needs Allan variance
  analysis
- No translational dynamics — deliberate, the stand is fixed
- The IMU model has no accelerometer bias, scale factor, axis misalignment or
  vibration; the motor thrust curve is linear and the servos have no backlash

## License

MIT — see [LICENSE](LICENSE).
