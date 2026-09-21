import numpy as np
import pytest
from sim.common.inertia import box_inertia, cylinder_inertia, parallel_axis_shift, combine_inertia

def test_box_cube_symmetry():
    I_cube = box_inertia(6.0, 2.0, 2.0, 2.0)
    assert np.isclose(I_cube[0,0], I_cube[1,1])
    assert np.isclose(I_cube[1,1], I_cube[2,2])

def test_cylinder_z_axis():
    I_cyl = cylinder_inertia(mass=2.0, radius=1.0, height=4.0, axis="z")
    assert np.isclose(I_cyl[2,2], 1.0)
    assert np.isclose(I_cyl[0,0], 3.16666667)

def test_cylinder_invalid_axis_raises():
    with pytest.raises(ValueError):
        cylinder_inertia(mass=1.0, radius=1.0, height=1.0, axis="q")

def test_parallel_axis_zero_offset():
    I_cm = box_inertia(6.0, 2.0, 2.0, 2.0)
    offset = np.array([0.0, 0.0, 0.0])
    I_shifted = parallel_axis_shift(I_cm, mass=3.0, offset=offset)
    assert np.allclose(I_shifted, I_cm)

def test_parallel_axis_x_offset_increases_perp_axes():
    I_test = box_inertia(1.0, 1.0, 1.0, 1.0)
    offset = np.array([2.0, 0.0, 0.0])
    I_shifted = parallel_axis_shift(I_test, mass=1.0, offset=offset)

    assert np.isclose(I_shifted[0, 0], I_test[0, 0])
    assert I_shifted[1, 1] > I_test[1, 1]
    assert I_shifted[2, 2] > I_test[2, 2]

def test_combine_inertia_sums_parts():
    part1 = np.eye(3) * 2.0
    part2 = np.eye(3) * 3.0
    parts = [part1, part2]

    result = combine_inertia(parts)
    expected = np.eye(3) * 5.0

    assert np.allclose(result, expected)
