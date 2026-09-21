import numpy as np

def build_allocation_matrix(arm_length, height, nominal_thrust,
                            include_yaw=True):
    rows = 3 if include_yaw else 2
    B = np.zeros((rows, 4))

    B[0, 0] = arm_length
    B[0, 1] = -arm_length

    B[1, 2] = -height * nominal_thrust
    B[1, 3] = -height * nominal_thrust

    if include_yaw:
        B[2, 2] = arm_length * nominal_thrust
        B[2, 3] = -arm_length * nominal_thrust

    return B


def pseudo_inverse(B):
    return B.T @ np.linalg.inv(B @ B.T)


def allocate(B_pinv, torque_desired):
    return B_pinv @ torque_desired


def saturate(u, max_thrust_delta, max_servo_angle):
    u_sat = u.copy()

    u_sat[0] = np.clip(u[0], -max_thrust_delta, max_thrust_delta)
    u_sat[1] = np.clip(u[1], -max_thrust_delta, max_thrust_delta)
    u_sat[2] = np.clip(u[2], -max_servo_angle, max_servo_angle)
    u_sat[3] = np.clip(u[3], -max_servo_angle, max_servo_angle)

    return u_sat

if __name__ == "__main__":
    L = 0.15
    h = 0.10
    T0 = 4.0

    B = build_allocation_matrix(L, h, T0)
    B_pinv = pseudo_inverse(B)

    print("B:\n", B)
    print("B_pinv:\n", B_pinv)
    print("det(B B^T):", np.linalg.det(B @ B.T))

    for label, tau in (("pure roll ", [0.3, 0.0, 0.0]),
                       ("pure pitch", [0.0, -0.1, 0.0]),
                       ("pure yaw  ", [0.0, 0.0, 0.1])):
        u = allocate(B_pinv, np.array(tau))
        print(f"\n{label} request -> u: {u}")
        print(f"            check B @ u: {B @ u}")

    print("\n--- saturation test ---")
    max_dT = 0.5
    max_servo = np.radians(10.0)

    tau_big = np.array([1.0, -0.5, 0.3])
    u_raw = allocate(B_pinv, tau_big)
    u_sat = saturate(u_raw, max_dT, max_servo)

    print("requested torque:", tau_big)
    print("raw command:   ", u_raw)
    print("clipped:       ", u_sat)
    print("actual torque: ", B @ u_sat)
