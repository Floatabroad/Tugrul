import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None, derivative_cutoff_hz=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.derivative_cutoff_hz = derivative_cutoff_hz

        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_filtered = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_filtered = 0.0

    def step(self, setpoint, measurement, dt, saturated=False):
        error = setpoint - measurement

        raw_derivative = (error - self.prev_error) / dt

        tau_d = 1.0 / (2 * np.pi * self.derivative_cutoff_hz)
        alpha = tau_d / (tau_d + dt)
        self.derivative_filtered = alpha * self.derivative_filtered + (1 - alpha) * raw_derivative

        derivative = self.derivative_filtered
        candidate_integral = self.integral + error * dt

        output = self.kp * error + self.ki * candidate_integral + self.kd * derivative

        clamped = output
        if self.output_limit is not None:
            clamped = np.clip(output, -self.output_limit, self.output_limit)

        if clamped == output and not saturated:
            self.integral = candidate_integral

        self.prev_error = error
        return clamped
