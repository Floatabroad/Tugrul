import numpy as np


class ServoModel:
    def __init__(self, slew_rate_deg=600.0, min_angle_deg=-30.0, max_angle_deg=30.0):
        self.slew_rate = np.radians(slew_rate_deg)
        self.min_angle = np.radians(min_angle_deg)
        self.max_angle = np.radians(max_angle_deg)
        self.angle = 0.0

    def step(self, command, dt):
        command = np.clip(command, self.min_angle, self.max_angle)

        max_step = self.slew_rate * dt
        error = command - self.angle
        step = np.clip(error, -max_step, max_step)

        self.angle = self.angle + step
        return self.angle


if __name__ == "__main__":
    servo = ServoModel(slew_rate_deg=600.0)

    dt = 0.001
    command = np.radians(30.0)

    for i in range(10):
        servo.step(command, dt)
    print("after 10 ms (deg):", np.degrees(servo.angle))

    for i in range(40):
        servo.step(command, dt)
    print("after 50 ms (deg):", np.degrees(servo.angle))

    for i in range(100):
        servo.step(command, dt)
    print("after 150 ms (deg):", np.degrees(servo.angle))
