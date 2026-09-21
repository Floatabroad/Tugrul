import numpy as np
from sim.common.quaternion import quat_derivative, quat_normalize, quat_to_rotmat
from sim.plant.integrators import rk4_step

class RigidBodyPlant:

    def __init__(self, I, initial_state, mass=0.0, com_offset=None,
                 gravity=9.81):
        self.I = I
        self.I_inv = np.linalg.inv(I)
        self.state = initial_state

        self.mass = mass
        self.com_offset = (np.zeros(3) if com_offset is None
                           else np.asarray(com_offset, dtype=float))
        self.gravity_world = np.array([0.0, 0.0, gravity])
        self._has_gravity = mass > 0.0 and np.any(self.com_offset != 0.0)

    def gravity_torque(self, q):
        if not self._has_gravity:
            return np.zeros(3)

        force_body = self.mass * (quat_to_rotmat(q).T @ self.gravity_world)
        return np.cross(self.com_offset, force_body)

    def state_derivative(self, x, torque):
        q = x[0:4]
        omega = x[4:7]

        q_dot = quat_derivative(q, omega)

        total_torque = torque + self.gravity_torque(q)

        I_omega = self.I @ omega
        gyroscopic = np.cross(omega, I_omega)
        omega_dot = self.I_inv @ (total_torque - gyroscopic)

        x_dot = np.concatenate([q_dot, omega_dot])
        return x_dot

    def step(self, torque, dt):
        self.state = rk4_step(self.state_derivative, self.state, dt, torque)

        q = self.state[0:4]
        omega = self.state[4:7]
        q_normalized = quat_normalize(q)
        self.state = np.concatenate([q_normalized, omega])

if __name__ == "__main__":
    I = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    plant = RigidBodyPlant(I, initial_state)

    torque = np.array([0.0, 0.0, 1.0]) 
    dt = 0.01
    steps = 100

    for i in range(steps):
        plant.step(torque, dt)

    total_time = dt * steps
    omega_z_analytic = torque[2] / I[2,2] * total_time

    print("simulated omega:", plant.state[4:7])
    print("analytic omega_z:", omega_z_analytic)
