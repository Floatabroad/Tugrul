import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque

from sim.sim_loop import SimLoop
from sim.common.frames import quat_from_euler
from scripts.view3d import View3D


WINDOW_SECONDS = 6.0
STEPS_PER_FRAME = 25
FRAME_INTERVAL_MS = 25


class LiveSim:
    def __init__(self):
        self.sim = SimLoop()
        self.t = 0.0
        self.running = True
        self.disturbance_steps_left = 0
        self.disturbance_axis = 0

        n = int(WINDOW_SECONDS / self.sim.dt / STEPS_PER_FRAME) * STEPS_PER_FRAME
        self.times = deque(maxlen=n)
        self.roll_true = deque(maxlen=n)
        self.roll_est = deque(maxlen=n)
        self.roll_sp_log = deque(maxlen=n)
        self.pitch_true = deque(maxlen=n)
        self.pitch_est = deque(maxlen=n)
        self.pitch_sp_log = deque(maxlen=n)
        self.yaw_true = deque(maxlen=n)
        self.yaw_est = deque(maxlen=n)
        self.yaw_sp_log = deque(maxlen=n)
        self.thrust1 = deque(maxlen=n)
        self.thrust2 = deque(maxlen=n)
        self.servo1 = deque(maxlen=n)
        self.servo2 = deque(maxlen=n)

        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.yaw_sp = 0.0
        self.last_disturbance = None

        self._build_figure()
        self.view3d = View3D()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("Gimbal - live simulation")

        gs = self.fig.add_gridspec(3, 1, left=0.08, right=0.97,
                                   top=0.96, bottom=0.42, hspace=0.35)
        self.ax_roll = self.fig.add_subplot(gs[0])
        self.ax_pitch = self.fig.add_subplot(gs[1])
        self.ax_act = self.fig.add_subplot(gs[2])

        self.ln_roll_sp, = self.ax_roll.plot([], [], 'r:', lw=1.5, label="target")
        self.ln_roll_true, = self.ax_roll.plot([], [], lw=1.5, label="true")
        self.ln_roll_est, = self.ax_roll.plot([], [], '--', lw=1.2, label="EKF")
        self.ax_roll.set_ylabel("roll (deg)")
        self.ax_roll.set_ylim(-25, 25)
        self.ax_roll.legend(loc="upper right", fontsize=8)
        self.ax_roll.grid(True, alpha=0.3)

        self.ln_pitch_sp, = self.ax_pitch.plot([], [], 'r:', lw=1.5, label="target")
        self.ln_pitch_true, = self.ax_pitch.plot([], [], lw=1.5, label="true")
        self.ln_pitch_est, = self.ax_pitch.plot([], [], '--', lw=1.2, label="EKF")
        self.ax_pitch.set_ylabel("pitch (deg)")
        self.ax_pitch.set_ylim(-25, 25)
        self.ax_pitch.legend(loc="upper right", fontsize=8)
        self.ax_pitch.grid(True, alpha=0.3)

        self.ln_t1, = self.ax_act.plot([], [], lw=1.2, label="motor 1 (N)")
        self.ln_t2, = self.ax_act.plot([], [], lw=1.2, label="motor 2 (N)")
        self.ln_s1, = self.ax_act.plot([], [], lw=1.2, label="servo 1 (deg)")
        self.ln_s2, = self.ax_act.plot([], [], lw=1.2, label="servo 2 (deg)")
        self.ax_act.set_ylabel("actuator")
        self.ax_act.set_xlabel("time (s)")
        self.ax_act.set_ylim(-25, 25)
        self.ax_act.legend(loc="upper right", fontsize=8, ncol=2)
        self.ax_act.grid(True, alpha=0.3)

        self._build_controls()

    def _build_controls(self):
        def slider(x, y, w, label, lo, hi, init, fmt="%.2f"):
            ax = self.fig.add_axes([x, y, w, 0.022])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        col1, col2 = 0.10, 0.62
        w = 0.28

        self.s_roll_sp = slider(col1, 0.33, w, "target roll", -20, 20, 0.0)
        self.s_pitch_sp = slider(col1, 0.29, w, "target pitch", -20, 20, 0.0)
        self.s_yaw_sp = slider(col2, 0.25, w, "target yaw", -45, 45, 0.0)

        self.s_kp_roll = slider(col1, 0.23, w, "kp roll", 0.0, 8.0, 2.0)
        self.s_ki_roll = slider(col1, 0.19, w, "ki roll", 0.0, 2.0, 0.1)
        self.s_kd_roll = slider(col1, 0.15, w, "kd roll", 0.0, 10.0, 3.0)
        self.s_fc_roll = slider(col1, 0.11, w, "filter roll (Hz)", 2.0, 200.0, 80.0, "%.0f")

        self.s_kp_pitch = slider(col2, 0.23, w, "kp pitch", 0.0, 3.0, 0.3)
        self.s_ki_pitch = slider(col2, 0.19, w, "ki pitch", 0.0, 0.5, 0.02, "%.3f")
        self.s_kd_pitch = slider(col2, 0.15, w, "kd pitch", 0.0, 3.0, 0.4)
        self.s_fc_pitch = slider(col2, 0.11, w, "filter pitch (Hz)", 2.0, 200.0, 15.0, "%.0f")

        self.s_kp_yaw = slider(col2, 0.07, w, "kp yaw", 0.0, 3.0, 0.5)
        self.s_ki_yaw = slider(col1, 0.07, w, "ki yaw", 0.0, 1.0, 0.05, "%.3f")
        self.s_noise = slider(col2, 0.33, w, "noise x", 0.1, 5.0, 1.0)
        self.s_dropout = slider(col2, 0.29, w, "dropout %", 0.0, 95.0, 0.0, "%.0f")

        def button(x, y, w_, label):
            ax = self.fig.add_axes([x, y, w_, 0.04])
            return Button(ax, label)

        self.b_pause = button(0.10, 0.03, 0.14, "Pause")
        self.b_dist_roll = button(0.27, 0.03, 0.16, "Impulse: roll")
        self.b_dist_pitch = button(0.46, 0.03, 0.16, "Impulse: pitch")
        self.b_reset = button(0.65, 0.03, 0.14, "Reset")
        self.b_zero = button(0.82, 0.03, 0.14, "Gains: default")

        self.b_pause.on_clicked(self._toggle_pause)
        self.b_dist_roll.on_clicked(lambda e: self._kick(0))
        self.b_dist_pitch.on_clicked(lambda e: self._kick(1))
        self.b_reset.on_clicked(self._reset)
        self.b_zero.on_clicked(self._default_gains)

        self.txt = self.fig.text(0.10, 0.375, "", fontsize=9, family="monospace")

    def _toggle_pause(self, event):
        self.running = not self.running
        self.b_pause.label.set_text("Resume" if not self.running else "Pause")

    def _kick(self, axis):
        self.disturbance_axis = axis
        self.disturbance_steps_left = 100

    def _reset(self, event):
        noise_mult = self.s_noise.val
        self.sim = SimLoop(gyro_noise_std=0.01 * noise_mult,
                           accel_noise_std=0.05 * noise_mult,
                           mag_noise_std=300.0 * noise_mult)
        self.t = 0.0
        for d in (self.times, self.roll_true, self.roll_est, self.roll_sp_log,
                  self.pitch_true, self.pitch_est, self.pitch_sp_log,
                  self.yaw_true, self.yaw_est, self.yaw_sp_log,
                  self.thrust1, self.thrust2, self.servo1, self.servo2):
            d.clear()

    def _default_gains(self, event):
        self.s_kp_roll.set_val(2.0)
        self.s_ki_roll.set_val(0.1)
        self.s_kd_roll.set_val(3.0)
        self.s_fc_roll.set_val(80.0)
        self.s_kp_pitch.set_val(0.3)
        self.s_ki_pitch.set_val(0.02)
        self.s_kd_pitch.set_val(0.4)
        self.s_fc_pitch.set_val(15.0)
        self.s_kp_yaw.set_val(0.5)
        self.s_ki_yaw.set_val(0.05)

    def _apply_gains(self):
        p = self.sim.pid_roll
        p.kp, p.ki, p.kd = self.s_kp_roll.val, self.s_ki_roll.val, self.s_kd_roll.val
        p.derivative_cutoff_hz = self.s_fc_roll.val

        p = self.sim.pid_pitch
        p.kp, p.ki, p.kd = self.s_kp_pitch.val, self.s_ki_pitch.val, self.s_kd_pitch.val
        p.derivative_cutoff_hz = self.s_fc_pitch.val

        p = self.sim.pid_yaw
        p.kp, p.ki = self.s_kp_yaw.val, self.s_ki_yaw.val

    def _advance(self):
        self._apply_gains()
        self.roll_sp = np.radians(self.s_roll_sp.val)
        self.pitch_sp = np.radians(self.s_pitch_sp.val)
        self.yaw_sp = np.radians(self.s_yaw_sp.val)

        dropout_rate = self.s_dropout.val / 100.0
        self.last_disturbance = None

        for _ in range(STEPS_PER_FRAME):
            disturbance = None
            if self.disturbance_steps_left > 0:
                d = np.zeros(3)
                d[self.disturbance_axis] = 0.4
                disturbance = d
                self.last_disturbance = d
                self.disturbance_steps_left -= 1

            dropout = np.random.random() < dropout_rate

            r = self.sim.step(self.roll_sp, self.pitch_sp, self.yaw_sp,
                              disturbance_torque=disturbance,
                              sensor_dropout=dropout)

            self.t += self.sim.dt
            self.times.append(self.t)
            self.roll_true.append(np.degrees(r["roll_true"]))
            self.roll_est.append(np.degrees(r["roll_est"]))
            self.roll_sp_log.append(np.degrees(self.roll_sp))
            self.pitch_true.append(np.degrees(r["pitch_true"]))
            self.pitch_est.append(np.degrees(r["pitch_est"]))
            self.pitch_sp_log.append(np.degrees(self.pitch_sp))
            self.yaw_true.append(np.degrees(r["yaw_true"]))
            self.yaw_est.append(np.degrees(r["yaw_est"]))
            self.yaw_sp_log.append(np.degrees(self.yaw_sp))
            self.thrust1.append(r["thrust"][0])
            self.thrust2.append(r["thrust"][1])
            self.servo1.append(np.degrees(r["servo"][0]))
            self.servo2.append(np.degrees(r["servo"][1]))

        return r

    def _redraw(self, last):
        t = np.array(self.times)
        if len(t) < 2:
            return

        self.ln_roll_sp.set_data(t, self.roll_sp_log)
        self.ln_roll_true.set_data(t, self.roll_true)
        self.ln_roll_est.set_data(t, self.roll_est)

        self.ln_pitch_sp.set_data(t, self.pitch_sp_log)
        self.ln_pitch_true.set_data(t, self.pitch_true)
        self.ln_pitch_est.set_data(t, self.pitch_est)

        self.ln_t1.set_data(t, self.thrust1)
        self.ln_t2.set_data(t, self.thrust2)
        self.ln_s1.set_data(t, self.servo1)
        self.ln_s2.set_data(t, self.servo2)

        for ax in (self.ax_roll, self.ax_pitch, self.ax_act):
            ax.set_xlim(t[0], max(t[-1], t[0] + 1.0))

        roll_err = self.roll_sp_log[-1] - self.roll_true[-1]
        pitch_err = self.pitch_sp_log[-1] - self.pitch_true[-1]
        yaw_err = self.yaw_sp_log[-1] - self.yaw_true[-1]
        ekf_err = abs(self.roll_true[-1] - self.roll_est[-1])

        self.txt.set_text(
            f"t={self.t:6.2f}s   roll err={roll_err:+6.2f}   "
            f"pitch err={pitch_err:+6.2f}   yaw err={yaw_err:+6.2f} deg   "
            f"EKF dev={ekf_err:5.3f} deg"
        )

        self.fig.canvas.draw_idle()

    def _update_3d(self, last):
        if self.view3d is None or not self.view3d.is_open():
            return

        q_target = quat_from_euler(self.roll_sp, self.pitch_sp, self.yaw_sp)
        hud = (f"t={self.t:5.2f}s\n"
               f"roll  {np.degrees(last['roll_true']):+6.2f} / "
               f"{np.degrees(self.roll_sp):+6.2f}\n"
               f"pitch {np.degrees(last['pitch_true']):+6.2f} / "
               f"{np.degrees(self.pitch_sp):+6.2f}\n"
               f"yaw   {np.degrees(last['yaw_true']):+6.2f} / "
               f"{np.degrees(self.yaw_sp):+6.2f}\n"
               f"T {last['thrust'][0]:4.2f} {last['thrust'][1]:4.2f} N\n"
               f"S {np.degrees(last['servo'][0]):+5.1f} "
               f"{np.degrees(last['servo'][1]):+5.1f} deg")

        self.view3d.update(self.sim.plant.state[0:4],
                           last["thrust"], last["servo"],
                           q_target=q_target,
                           disturbance=self.last_disturbance,
                           hud_text=hud)
        self.view3d.draw()

    def _tick(self):
        if self.running:
            last = self._advance()
            self._redraw(last)
            self._update_3d(last)

    def run(self):
        timer = self.fig.canvas.new_timer(interval=FRAME_INTERVAL_MS)
        timer.add_callback(self._tick)
        timer.start()
        plt.show()


if __name__ == "__main__":
    LiveSim().run()
