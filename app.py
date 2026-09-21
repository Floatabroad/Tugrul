#!/usr/bin/env python3
import sys

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def _live_sim():
    from scripts.live_sim import LiveSim
    LiveSim().run()


def _sandbox():
    from scripts.sandbox import Sandbox
    Sandbox().run()


def _tuner():
    from scripts.tuner import Tuner
    Tuner()
    plt.show()


def _view3d():
    from scripts.view3d import _demo
    _demo()


def _full_sim():
    from scripts.run_full_sim import main
    main()


def _ekf_test():
    from scripts.run_ekf_test import main
    main()


def _pid_test():
    from scripts.run_pid_test import main
    main()


def _baseline():
    from scripts.check_baseline import main
    main()
    _pause()


def _stress():
    from scripts import run_monte_carlo as mc

    sweeps = [
        ("Monte Carlo (100 runs, ~1-2 min)", lambda: mc.monte_carlo(n_runs=100)),
        ("Sensor dropout sweep", mc.dropout_sweep),
        ("Timing jitter sweep", mc.jitter_sweep),
        ("Loop stall sweep", mc.stall_sweep),
        ("Disturbance torque sweep", mc.disturbance_sweep),
    ]

    print("\n--- Stress tests ---")
    for i, (label, _) in enumerate(sweeps, 1):
        print(f"  {i}) {label}")
    print("  0) back")

    choice = input("\nSelect: ").strip()
    if choice == "0" or not choice:
        return
    try:
        _, fn = sweeps[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    print("\nRunning, please wait...\n")
    fn()
    _pause()


MODES = {
    "1": ("live", "Live simulation (sliders + 3D window)", _live_sim),
    "2": ("sandbox", "Torque sandbox (right-click drag to push)", _sandbox),
    "3": ("tuner", "Tuning tool (compare two gain sets side by side)", _tuner),
    "4": ("view3d", "3D visualisation demo", _view3d),
    "5": ("full", "Full closed-loop run + plots", _full_sim),
    "6": ("ekf", "EKF error + gyro bias plots", _ekf_test),
    "7": ("pid", "Single-axis PID step response", _pid_test),
    "8": ("stress", "Stress tests (Monte Carlo, dropout, stall...)", _stress),
    "9": ("baseline", "Regression check (deterministic)", _baseline),
}

ALIASES = {name: key for key, (name, _, _) in MODES.items()}


def _pause():
    try:
        input("\nPress Enter to return to the menu...")
    except EOFError:
        pass


def _run(key):
    _, label, fn = MODES[key]
    print(f"\n>>> {label}\n")
    try:
        fn()
    except KeyboardInterrupt:
        print("\n(interrupted)")
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        _pause()
    finally:
        plt.close("all")


def _banner():
    print("=" * 62)
    print("  Dual-EDF Gimbal Thrust Stand — simulation")
    print("=" * 62)


def _menu():
    while True:
        _banner()
        for key, (name, label, _) in MODES.items():
            print(f"  {key:>2}) {label}")
        print("   0) quit")

        try:
            choice = input("\nSelect: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice in ("0", "q", "quit", "exit"):
            return 0
        if choice in ALIASES:
            choice = ALIASES[choice]
        if choice in MODES:
            _run(choice)
        else:
            print("\nInvalid selection.\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv:
        arg = argv[0].lstrip("-").lower()
        if arg in ("help", "h", "list"):
            _banner()
            print("\nUsage: gimbal-sim [mode]\n\nModes:")
            for _, (name, label, _) in MODES.items():
                print(f"  {name:<9} {label}")
            return 0
        key = ALIASES.get(arg, arg if arg in MODES else None)
        if key is None:
            print(f"Unknown mode: {argv[0]}  (see --help for the list)")
            return 1
        _run(key)
        return 0

    return _menu()


if __name__ == "__main__":
    sys.exit(main())
