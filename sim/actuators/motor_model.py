import numpy as np


class MotorModel:
    def __init__(self, tau=0.03, max_thrust=10.0):
        self.tau = tau
        self.max_thrust = max_thrust
        self.thrust = 0.0

    def step(self, command, dt):
        command = np.clip(command, 0.0, self.max_thrust)
        alpha = 1.0 - np.exp(-dt / self.tau)
        self.thrust = self.thrust + (command - self.thrust) * alpha
        self.thrust = float(np.clip(self.thrust, 0.0, self.max_thrust))
        return self.thrust

if __name__ == "__main__":
    motor = MotorModel(tau=0.03, max_thrust=10.0)

    dt = 0.001
    command = 5.0

    for i in range(int(0.03 / dt)):
        motor.step(command, dt)
    print("after 1 tau:", motor.thrust)

    for i in range(int(0.06 / dt)):
        motor.step(command, dt)
    print("after 3 tau:", motor.thrust)

    for i in range(int(0.5 / dt)):
        motor.step(command, dt)
    print("after a long time:", motor.thrust)
