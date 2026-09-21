import numpy as np
def box_inertia(mass: float, dx: float, dy: float, dz: float) -> np.ndarray:
    Ixx = (mass/12) * (dy**2 + dz**2)
    Iyy = (mass/12) * (dx**2 + dz**2)
    Izz = (mass/12) * (dx**2 + dy**2)
    return np.array([
        [Ixx, 0, 0],
        [0, Iyy, 0],
        [0, 0, Izz]
    ])
def cylinder_inertia(mass: float, radius: float, height: float, axis: str = "z") -> np.ndarray:
    I_spin = 0.5 * mass * radius**2
    I_perp = (1/12) * mass * (3*radius**2 + height**2)
    if axis == "z":
        return np.array([
            [I_perp, 0, 0],
            [0, I_perp, 0],
            [0, 0, I_spin]
            ])
    elif axis == "x":
        return np.array([
            [I_spin, 0, 0],
            [0, I_perp, 0],
            [0, 0, I_perp]
            ])
    elif axis == "y":
        return np.array([
            [I_perp, 0, 0],
            [0, I_spin, 0],
            [0, 0, I_perp]
            ])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    
def parallel_axis_shift(I_cm: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    d_squared = np.dot(offset, offset)
    outer = np.outer(offset, offset)
    identity_3x3 = np.eye(3)
    I_shifted = I_cm + mass *(d_squared * identity_3x3 - outer)
    return I_shifted
def combine_inertia(parts: list[np.ndarray]) -> np.ndarray:
    total = np.zeros((3, 3))
    for part in parts:
        total = total + part
    return total
