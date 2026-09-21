import numpy as np
import matplotlib.pyplot as plt

from sim.common.quaternion import quat_to_rotmat
from sim.common.frames import quat_from_euler


BOX_HALF = np.array([0.05, 0.18, 0.03])
ARM_LENGTH = 0.15
EDF_HEIGHT = 0.10
THRUST_SCALE = 0.035
AXIS_LENGTH = 0.20
VIEW_LIMIT = 0.45

EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def box_vertices(half_extents):
    hx, hy, hz = half_extents
    return np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])


def edf_force_direction(servo_angle):
    return np.array([np.sin(servo_angle), 0.0, -np.cos(servo_angle)])


class View3D:

    def __init__(self, title="Gimbal - 3D"):
        self.fig = plt.figure(figsize=(7.5, 7.5))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        self.ax = self.fig.add_subplot(projection='3d')
        self.verts = box_vertices(BOX_HALF)

        self.edf_local = np.array([
            [0.0, -ARM_LENGTH, -EDF_HEIGHT],
            [0.0, +ARM_LENGTH, -EDF_HEIGHT],
        ])

        self._init_artists()
        self._style_axes()

    def _init_artists(self):
        self.ghost_lines = [
            self.ax.plot([], [], [], color='0.8', lw=1.0)[0]
            for _ in EDGES
        ]
        self.body_lines = [
            self.ax.plot([], [], [], color='#1f77b4', lw=2.0)[0]
            for _ in EDGES
        ]
        self.arm_lines = [
            self.ax.plot([], [], [], color='#444444', lw=2.5)[0]
            for _ in range(2)
        ]
        self.edf_dots = [
            self.ax.plot([], [], [], 'o', color='#333333', ms=7)[0]
            for _ in range(2)
        ]
        self.thrust_lines = [
            self.ax.plot([], [], [], color='#d62728', lw=3.0)[0]
            for _ in range(2)
        ]
        self.thrust_tips = [
            self.ax.plot([], [], [], '^', color='#d62728', ms=6)[0]
            for _ in range(2)
        ]

        axis_colors = ['#cc0000', '#00aa00', '#0000cc']
        self.axis_lines = [
            self.ax.plot([], [], [], color=c, lw=2.0)[0]
            for c in axis_colors
        ]

        self.disturb_line, = self.ax.plot([], [], [], color='#ff8800', lw=4.0)

        self.hud = self.ax.text2D(0.02, 0.96, "", transform=self.ax.transAxes,
                                  family="monospace", fontsize=9,
                                  verticalalignment='top')

    def _style_axes(self):
        self.ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
        self.ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)
        self.ax.set_zlim(VIEW_LIMIT, -VIEW_LIMIT)
        self.ax.set_xlabel("x (forward)")
        self.ax.set_ylabel("y (right)")
        self.ax.set_zlabel("z (down)")
        self.ax.view_init(elev=22, azim=-56)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    @staticmethod
    def _set_seg(line, p0, p1):
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    @staticmethod
    def _set_point(artist, p):
        artist.set_data([p[0]], [p[1]])
        artist.set_3d_properties([p[2]])

    def _draw_box(self, lines, verts_world):
        for line, (i, j) in zip(lines, EDGES):
            self._set_seg(line, verts_world[i], verts_world[j])

    def _clear_box(self, lines):
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

    def update(self, q_true, thrust, servo, q_target=None,
               disturbance=None, hud_text=""):
        R = quat_to_rotmat(q_true)

        body = self.verts @ R.T
        self._draw_box(self.body_lines, body)

        if q_target is not None:
            Rt = quat_to_rotmat(q_target)
            self._draw_box(self.ghost_lines, self.verts @ Rt.T)
        else:
            self._clear_box(self.ghost_lines)

        origin = np.zeros(3)
        edf_world = self.edf_local @ R.T

        for k in range(2):
            self._set_seg(self.arm_lines[k], origin, edf_world[k])
            self._set_point(self.edf_dots[k], edf_world[k])

            f_body = edf_force_direction(servo[k]) * thrust[k] * THRUST_SCALE
            f_world = R @ f_body
            tip = edf_world[k] + f_world

            self._set_seg(self.thrust_lines[k], edf_world[k], tip)
            self._set_point(self.thrust_tips[k], tip)

        for i in range(3):
            self._set_seg(self.axis_lines[i], origin, R[:, i] * AXIS_LENGTH)

        if disturbance is not None and np.linalg.norm(disturbance) > 1e-9:
            d = disturbance / np.linalg.norm(disturbance) * AXIS_LENGTH * 1.4
            self._set_seg(self.disturb_line, origin, d)
        else:
            self.disturb_line.set_data([], [])
            self.disturb_line.set_3d_properties([])

        self.hud.set_text(hud_text)

    def draw(self):
        self.fig.canvas.draw_idle()

    def is_open(self):
        return plt.fignum_exists(self.fig.number)


def _demo():
    import matplotlib
    matplotlib.use('TkAgg')
    from sim.sim_loop import SimLoop

    sim = SimLoop()
    view = View3D("Gimbal - 3D (demo)")

    state = {"t": 0.0}
    steps_per_frame = 25

    def tick():
        if not view.is_open():
            return

        t = state["t"]
        roll_sp = np.radians(8.0 * np.sin(2 * np.pi * 0.2 * t))
        pitch_sp = np.radians(5.0 * np.sin(2 * np.pi * 0.13 * t))
        yaw_sp = np.radians(15.0 * np.sin(2 * np.pi * 0.07 * t))

        disturbance = None
        if 6.0 <= t < 6.1:
            disturbance = np.array([0.4, 0.0, 0.0])

        r = None
        for _ in range(steps_per_frame):
            r = sim.step(roll_sp, pitch_sp, yaw_sp, disturbance_torque=disturbance)
            state["t"] += sim.dt

        q_target = quat_from_euler(roll_sp, pitch_sp, yaw_sp)
        hud = (f"t={state['t']:5.2f}s\n"
               f"roll  {np.degrees(r['roll_true']):+6.2f} / "
               f"{np.degrees(roll_sp):+6.2f}\n"
               f"pitch {np.degrees(r['pitch_true']):+6.2f} / "
               f"{np.degrees(pitch_sp):+6.2f}\n"
               f"yaw   {np.degrees(r['yaw_true']):+6.2f} / "
               f"{np.degrees(yaw_sp):+6.2f}\n"
               f"T {r['thrust'][0]:4.2f} {r['thrust'][1]:4.2f} N\n"
               f"S {np.degrees(r['servo'][0]):+5.1f} "
               f"{np.degrees(r['servo'][1]):+5.1f} deg")

        view.update(sim.plant.state[0:4], r["thrust"], r["servo"],
                    q_target=q_target, disturbance=disturbance, hud_text=hud)
        view.draw()

    timer = view.fig.canvas.new_timer(interval=33)
    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    _demo()
