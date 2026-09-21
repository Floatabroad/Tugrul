import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from sim.sim_loop import SimLoop
from sim.common.frames import quat_from_euler
from scripts.view3d import View3D

plt.rcParams['keymap.save'] = []
STEPS_PER_FRAME = 25
FRAME_INTERVAL_MS = 25
IMPULSE_STEPS = 120
IMPULSE_MAGNITUDE = 0.5
TRAIL_LENGTH = 300


class Sandbox:
    def __init__(self):
        self.sim = SimLoop()
        self.t = 0.0
        self.running = True
        self.controller_on = True
        self.torque_gain = 0.004

        self.drag_torque = np.zeros(3)
        self.drag_origin = None
        self.impulse = np.zeros(3)
        self.impulse_steps_left = 0

        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.yaw_sp = 0.0

        self.trail = deque(maxlen=TRAIL_LENGTH)

        p = self.sim.pid_roll
        self.gains_roll = (p.kp, p.ki, p.kd)
        p = self.sim.pid_pitch
        self.gains_pitch = (p.kp, p.ki, p.kd)
        p = self.sim.pid_yaw
        self.gains_yaw = (p.kp, p.ki, p.kd)

        self.view = View3D("Torque sandbox - right-drag to push")
        self._add_trail_artist()
        self._connect_events()

    def _add_trail_artist(self):
        ax = self.view.ax
        self.trail_line, = ax.plot([], [], [], color='#9467bd', lw=1.0, alpha=0.7)
        self.help_text = ax.text2D(
            0.02, 0.02,
            "right-drag: push   |   WASD: roll/pitch, QE: yaw impulse\n"
            "space: pause   |   R: reset   |   +/-: sensitivity",
            transform=ax.transAxes, fontsize=8, family="monospace",
            color='0.35', verticalalignment='bottom')
        self.status_text = ax.text2D(
            0.98, 0.96, "", transform=ax.transAxes, fontsize=10,
            family="monospace", horizontalalignment='right',
            verticalalignment='top')

    def _update_trail(self, R):
        tip = R[:, 2] * -0.20
        self.trail.append(tip)

        if len(self.trail) > 2:
            arr = np.array(self.trail)
            self.trail_line.set_data(arr[:, 0], arr[:, 1])
            self.trail_line.set_3d_properties(arr[:, 2])

    def _connect_events(self):
        c = self.view.fig.canvas
        c.mpl_connect('button_press_event', self._on_press)
        c.mpl_connect('motion_notify_event', self._on_motion)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('key_press_event', self._on_key)

    def _on_press(self, event):
        if event.button == 3 and event.inaxes is self.view.ax:
            self.drag_origin = (event.x, event.y)
            self.drag_torque = np.zeros(3)

    def _on_motion(self, event):
        if self.drag_origin is None or event.x is None:
            return
        dx = event.x - self.drag_origin[0]
        dy = event.y - self.drag_origin[1]
        self.drag_torque = np.array([
            dx * self.torque_gain,
            -dy * self.torque_gain,
            0.0,
        ])

    def _on_release(self, event):
        if event.button == 3:
            self.drag_origin = None
            self.drag_torque = np.zeros(3)

    def _on_key(self, event):
        k = event.key

        if k == 'a':
            self._kick(np.array([-IMPULSE_MAGNITUDE, 0.0, 0.0]))
        elif k == 'd':
            self._kick(np.array([IMPULSE_MAGNITUDE, 0.0, 0.0]))
        elif k == 'w':
            self._kick(np.array([0.0, IMPULSE_MAGNITUDE, 0.0]))
        elif k == 's':
            self._kick(np.array([0.0, -IMPULSE_MAGNITUDE, 0.0]))
        elif k == 'q':
            self._kick(np.array([0.0, 0.0, -IMPULSE_MAGNITUDE]))
        elif k == 'e':
            self._kick(np.array([0.0, 0.0, IMPULSE_MAGNITUDE]))
        elif k == 'c':
            self.controller_on = not self.controller_on
        elif k == ' ':
            self.running = not self.running
        elif k == 'r':
            self._reset()
        elif k in ('+', '='):
            self.torque_gain = min(self.torque_gain * 1.5, 0.05)
        elif k == '-':
            self.torque_gain = max(self.torque_gain / 1.5, 0.0005)

    def _kick(self, torque):
        self.impulse = torque
        self.impulse_steps_left = IMPULSE_STEPS

    def _reset(self):
        self.sim = SimLoop()
        self.t = 0.0
        self.trail.clear()
        self.impulse_steps_left = 0
        self.drag_torque = np.zeros(3)

    def _external_torque(self):
        total = self.drag_torque.copy()
        if self.impulse_steps_left > 0:
            total = total + self.impulse
            self.impulse_steps_left -= 1
        return total if np.linalg.norm(total) > 1e-9 else None

    def _advance(self):
        self._apply_controller_state()

        r = None
        ext = None
        for _ in range(STEPS_PER_FRAME):
            ext = self._external_torque()
            r = self.sim.step(self.roll_sp, self.pitch_sp, self.yaw_sp,
                              disturbance_torque=ext)
            self.t += self.sim.dt

        return r, ext

    def _apply_controller_state(self):
        for pid, gains in ((self.sim.pid_roll, self.gains_roll),
                           (self.sim.pid_pitch, self.gains_pitch),
                           (self.sim.pid_yaw, self.gains_yaw)):
            if self.controller_on:
                pid.kp, pid.ki, pid.kd = gains
            else:
                pid.kp = pid.ki = pid.kd = 0.0
                pid.integral = 0.0

    def _tick(self):
        if not self.view.is_open():
            return
        if not self.running:
            return

        r, ext = self._advance()

        from sim.common.quaternion import quat_to_rotmat
        R = quat_to_rotmat(self.sim.plant.state[0:4])
        self._update_trail(R)

        mag = 0.0 if ext is None else float(np.linalg.norm(ext))

        hud = (f"t={self.t:6.2f}s\n"
               f"roll  {np.degrees(r['roll_true']):+7.2f} deg\n"
               f"pitch {np.degrees(r['pitch_true']):+7.2f} deg\n"
               f"yaw   {np.degrees(r['yaw_true']):+7.2f} deg\n"
               f"T {r['thrust'][0]:4.2f} {r['thrust'][1]:4.2f} N\n"
               f"S {np.degrees(r['servo'][0]):+5.1f} "
               f"{np.degrees(r['servo'][1]):+5.1f} deg\n"
               f"ext torque {mag:5.3f} N.m")

        status = ("CONTROLLER ON" if self.controller_on
                  else "CONTROLLER OFF")
        color = '#2ca02c' if self.controller_on else '#d62728'
        self.status_text.set_text(status)
        self.status_text.set_color(color)

        q_target = quat_from_euler(self.roll_sp, self.pitch_sp, self.yaw_sp)
        self.view.update(self.sim.plant.state[0:4],
                         r["thrust"], r["servo"],
                         q_target=q_target,
                         disturbance=ext,
                         hud_text=hud)
        self.view.draw()

    def run(self):
        timer = self.view.fig.canvas.new_timer(interval=FRAME_INTERVAL_MS)
        timer.add_callback(self._tick)
        timer.start()
        plt.show()


if __name__ == "__main__":
    Sandbox().run()
