import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from sim.sim_loop import SimLoop


DURATION = 8.0


def step_profile(t, amplitude_deg):
    return np.radians(amplitude_deg) if t > 0.5 else 0.0


def sine_profile(t, amplitude_deg):
    return np.radians(amplitude_deg * np.sin(2 * np.pi * 0.3 * t))


PROFILES = {
    "step": step_profile,
    "sine": sine_profile,
}


class Tuner:
    def __init__(self):
        self.previous = None
        self.profile_name = "step"
        self._build_figure()
        self.run_once(None)

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("Gimbal - tuning")

        gs = self.fig.add_gridspec(4, 1, left=0.08, right=0.97,
                                   top=0.96, bottom=0.44, hspace=0.40)
        self.ax_roll = self.fig.add_subplot(gs[0])
        self.ax_pitch = self.fig.add_subplot(gs[1])
        self.ax_yaw = self.fig.add_subplot(gs[2])
        self.ax_act = self.fig.add_subplot(gs[3])

        for ax, name in ((self.ax_roll, "roll"), (self.ax_pitch, "pitch"),
                         (self.ax_yaw, "yaw")):
            ax.set_ylabel(f"{name} (deg)")
            ax.grid(True, alpha=0.3)
        self.ax_act.set_ylabel("actuator")
        self.ax_act.set_xlabel("time (s)")
        self.ax_act.grid(True, alpha=0.3)

        self._build_controls()

    def _build_controls(self):
        def slider(x, y, w, label, lo, hi, init, fmt="%.2f"):
            ax = self.fig.add_axes([x, y, w, 0.022])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        col1, col2 = 0.10, 0.62
        w = 0.28

        self.s_kp_roll = slider(col1, 0.33, w, "kp roll", 0.0, 8.0, 2.0)
        self.s_ki_roll = slider(col1, 0.29, w, "ki roll", 0.0, 2.0, 0.1)
        self.s_kd_roll = slider(col1, 0.25, w, "kd roll", 0.0, 10.0, 3.0)
        self.s_fc_roll = slider(col1, 0.21, w, "filter roll (Hz)", 2.0, 200.0, 80.0, "%.0f")
        self.s_lim_roll = slider(col1, 0.17, w, "limit roll", 0.05, 2.0, 1.0)

        self.s_kp_pitch = slider(col2, 0.33, w, "kp pitch", 0.0, 3.0, 0.3)
        self.s_ki_pitch = slider(col2, 0.29, w, "ki pitch", 0.0, 0.5, 0.02, "%.3f")
        self.s_kd_pitch = slider(col2, 0.25, w, "kd pitch", 0.0, 3.0, 0.4)
        self.s_fc_pitch = slider(col2, 0.21, w, "filter pitch (Hz)", 2.0, 200.0, 15.0, "%.0f")
        self.s_lim_pitch = slider(col2, 0.17, w, "limit pitch", 0.05, 2.0, 0.3)

        self.s_kp_yaw = slider(col1, 0.13, w, "kp yaw", 0.0, 3.0, 0.5)
        self.s_ki_yaw = slider(col2, 0.13, w, "ki yaw", 0.0, 1.0, 0.05, "%.3f")
        self.s_kd_yaw = slider(col1, 0.09, w, "kd yaw", 0.0, 3.0, 0.5)
        self.s_lim_yaw = slider(col2, 0.09, w, "limit yaw", 0.05, 1.0, 0.4)

        self.s_roll_amp = slider(col1, 0.05, w, "target roll", -15, 15, 5.0)
        self.s_pitch_amp = slider(col2, 0.05, w, "target pitch", -15, 15, 0.0)
        self.s_yaw_amp = slider(col1, 0.01, w, "target yaw", -45, 45, 0.0)

        ax_radio = self.fig.add_axes([0.62, 0.005, 0.07, 0.035])
        self.radio = RadioButtons(ax_radio, ("step", "sine"))
        self.radio.on_clicked(self._set_profile)

        def button(x, y, w_, label):
            ax = self.fig.add_axes([x, y, w_, 0.04])
            return Button(ax, label)

        self.b_run = button(0.70, 0.005, 0.09, "Run")
        self.b_dist = button(0.80, 0.005, 0.09, "+ impulse")
        self.b_clear = button(0.90, 0.005, 0.09, "Clear")
        self.b_default = button(0.62, 0.045, 0.37, "Gains: default")

        self.b_run.on_clicked(self.run_once)
        self.b_dist.on_clicked(lambda e: self.run_once(e, disturbance=True))
        self.b_clear.on_clicked(self._clear_history)
        self.b_default.on_clicked(self._default_gains)

        self.txt = self.fig.text(0.08, 0.405, "", fontsize=8, family="monospace")

    def _set_profile(self, label):
        self.profile_name = label

    def _clear_history(self, event):
        self.previous = None
        self._draw()

    def _default_gains(self, event):
        self.s_kp_roll.set_val(2.0)
        self.s_ki_roll.set_val(0.1)
        self.s_kd_roll.set_val(3.0)
        self.s_fc_roll.set_val(80.0)
        self.s_lim_roll.set_val(1.0)
        self.s_kp_pitch.set_val(0.3)
        self.s_ki_pitch.set_val(0.02)
        self.s_kd_pitch.set_val(0.4)
        self.s_fc_pitch.set_val(15.0)
        self.s_lim_pitch.set_val(0.3)
        self.s_kp_yaw.set_val(0.5)
        self.s_ki_yaw.set_val(0.05)
        self.s_kd_yaw.set_val(0.5)
        self.s_lim_yaw.set_val(0.4)

    def _simulate(self, disturbance=False):
        sim = SimLoop()

        p = sim.pid_roll
        p.kp, p.ki, p.kd = self.s_kp_roll.val, self.s_ki_roll.val, self.s_kd_roll.val
        p.derivative_cutoff_hz = self.s_fc_roll.val
        p.output_limit = self.s_lim_roll.val

        p = sim.pid_pitch
        p.kp, p.ki, p.kd = self.s_kp_pitch.val, self.s_ki_pitch.val, self.s_kd_pitch.val
        p.derivative_cutoff_hz = self.s_fc_pitch.val
        p.output_limit = self.s_lim_pitch.val

        p = sim.pid_yaw
        p.kp, p.ki, p.kd = self.s_kp_yaw.val, self.s_ki_yaw.val, self.s_kd_yaw.val
        p.output_limit = self.s_lim_yaw.val

        profile = PROFILES[self.profile_name]
        roll_amp = self.s_roll_amp.val
        pitch_amp = self.s_pitch_amp.val
        yaw_amp = self.s_yaw_amp.val

        steps = int(DURATION / sim.dt)
        log = {k: [] for k in ("t", "roll_sp", "roll_true", "roll_est",
                               "pitch_sp", "pitch_true", "pitch_est",
                               "yaw_sp", "yaw_true", "yaw_est",
                               "t1", "t2", "s1", "s2")}

        for i in range(steps):
            t = i * sim.dt

            dist = None
            if disturbance and 4.0 <= t < 4.1:
                dist = np.array([0.4, 0.0, 0.0])

            roll_sp = profile(t, roll_amp)
            pitch_sp = profile(t, pitch_amp)
            yaw_sp = profile(t, yaw_amp)

            r = sim.step(roll_sp, pitch_sp, yaw_sp, disturbance_torque=dist)

            log["t"].append(t)
            log["roll_sp"].append(np.degrees(roll_sp))
            log["roll_true"].append(np.degrees(r["roll_true"]))
            log["roll_est"].append(np.degrees(r["roll_est"]))
            log["pitch_sp"].append(np.degrees(pitch_sp))
            log["pitch_true"].append(np.degrees(r["pitch_true"]))
            log["pitch_est"].append(np.degrees(r["pitch_est"]))
            log["yaw_sp"].append(np.degrees(yaw_sp))
            log["yaw_true"].append(np.degrees(r["yaw_true"]))
            log["yaw_est"].append(np.degrees(r["yaw_est"]))
            log["t1"].append(r["thrust"][0])
            log["t2"].append(r["thrust"][1])
            log["s1"].append(np.degrees(r["servo"][0]))
            log["s2"].append(np.degrees(r["servo"][1]))

        for k in log:
            log[k] = np.array(log[k])

        return log

    @staticmethod
    def _metrics(log):
        t = log["t"]
        settle_mask = t > DURATION - 1.0

        roll_err = log["roll_sp"] - log["roll_true"]
        pitch_err = log["pitch_sp"] - log["pitch_true"]
        yaw_err = log["yaw_sp"] - log["yaw_true"]
        ekf_err = np.abs(log["roll_true"] - log["roll_est"])

        sp = log["roll_sp"]
        target = sp[-1]
        if abs(target) > 1e-6:
            overshoot = (np.max(log["roll_true"] * np.sign(target))
                         - abs(target)) / abs(target) * 100.0
            overshoot = max(overshoot, 0.0)
        else:
            overshoot = 0.0

        servo_activity = np.mean(np.abs(np.diff(log["s1"])))

        return {
            "roll_rms": float(np.sqrt(np.mean(roll_err[settle_mask] ** 2))),
            "pitch_rms": float(np.sqrt(np.mean(pitch_err[settle_mask] ** 2))),
            "yaw_rms": float(np.sqrt(np.mean(yaw_err[settle_mask] ** 2))),
            "ekf_mean": float(np.mean(ekf_err)),
            "overshoot": float(overshoot),
            "servo_activity": float(servo_activity),
        }

    def run_once(self, event, disturbance=False):
        if hasattr(self, "current"):
            self.previous = self.current
        self.current = self._simulate(disturbance=disturbance)
        self._draw()

    def _draw(self):
        for ax in (self.ax_roll, self.ax_pitch, self.ax_yaw, self.ax_act):
            ax.clear()
            ax.grid(True, alpha=0.3)

        if self.previous is not None:
            p = self.previous
            self.ax_roll.plot(p["t"], p["roll_true"], color="0.75", lw=1.0,
                              label="previous")
            self.ax_pitch.plot(p["t"], p["pitch_true"], color="0.75", lw=1.0,
                               label="previous")
            self.ax_yaw.plot(p["t"], p["yaw_true"], color="0.75", lw=1.0,
                             label="previous")
            self.ax_act.plot(p["t"], p["s1"], color="0.85", lw=1.0)

        c = self.current

        self.ax_roll.plot(c["t"], c["roll_sp"], 'r:', lw=1.5, label="target")
        self.ax_roll.plot(c["t"], c["roll_true"], lw=1.5, label="true")
        self.ax_roll.plot(c["t"], c["roll_est"], '--', lw=1.0, label="EKF")
        self.ax_roll.set_ylabel("roll (deg)")
        self.ax_roll.legend(loc="upper right", fontsize=8, ncol=4)

        self.ax_pitch.plot(c["t"], c["pitch_sp"], 'r:', lw=1.5, label="target")
        self.ax_pitch.plot(c["t"], c["pitch_true"], lw=1.5, label="true")
        self.ax_pitch.plot(c["t"], c["pitch_est"], '--', lw=1.0, label="EKF")
        self.ax_pitch.set_ylabel("pitch (deg)")
        self.ax_pitch.legend(loc="upper right", fontsize=8, ncol=4)

        self.ax_yaw.plot(c["t"], c["yaw_sp"], 'r:', lw=1.5, label="target")
        self.ax_yaw.plot(c["t"], c["yaw_true"], lw=1.5, label="true")
        self.ax_yaw.plot(c["t"], c["yaw_est"], '--', lw=1.0, label="EKF")
        self.ax_yaw.set_ylabel("yaw (deg)")
        self.ax_yaw.legend(loc="upper right", fontsize=8, ncol=4)

        self.ax_act.plot(c["t"], c["t1"], lw=1.0, label="motor 1 (N)")
        self.ax_act.plot(c["t"], c["t2"], lw=1.0, label="motor 2 (N)")
        self.ax_act.plot(c["t"], c["s1"], lw=1.0, label="servo 1 (deg)")
        self.ax_act.plot(c["t"], c["s2"], lw=1.0, label="servo 2 (deg)")
        self.ax_act.set_ylabel("actuator")
        self.ax_act.set_xlabel("time (s)")
        self.ax_act.legend(loc="upper right", fontsize=8, ncol=4)

        m = self._metrics(c)
        line = (f"roll RMS={m['roll_rms']:6.3f}  pitch RMS={m['pitch_rms']:6.3f}  "
                f"yaw RMS={m['yaw_rms']:6.3f}  EKF dev={m['ekf_mean']:5.3f}  "
                f"overshoot={m['overshoot']:5.1f}%  "
                f"servo act={m['servo_activity']:6.4f}")

        if self.previous is not None:
            mp = self._metrics(self.previous)
            line += (f"\nprevious: {mp['roll_rms']:6.3f}            "
                     f"{mp['pitch_rms']:6.3f}          {mp['yaw_rms']:6.3f}"
                     f"            {mp['ekf_mean']:5.3f}"
                     f"           {mp['overshoot']:5.1f}%"
                     f"            {mp['servo_activity']:6.4f}")

        self.txt.set_text(line)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Tuner()
    plt.show()
