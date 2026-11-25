"""Tests of `um_bilinear_weights`, the default method, on small hand-built grids.

Each test checks one rule of the UM reconfiguration's interpolation of land fields (see the README):
interior points (four valid corners) are interpolated bilinearly, coastal points (one to three) copy
their heaviest valid corner, and unresolved points (none) copy the nearest resolved target point.
Further tests cover tie-breaking, wrapping around the globe, clamping beyond the source's edges and
rotated grids.
"""

import numpy as np
import pytest
from conftest import (
    global_mask_cube,
    mask_cube,
    regrid_values,
    regular_mask_cube,
    true_latitudes,
    true_longitudes,
    with_mask,
)

from um_masked_regrid import (
    bilinear_corners,
    build_regridder,
    horizontal_coordinates,
    rotate_pole,
    um_bilinear_weights,
)

# A small regional source grid: rows at 0, 10, 20 N and columns at 0, 10, 20, 30 E.
REGIONAL_LATITUDES = [0.0, 10.0, 20.0]
REGIONAL_LONGITUDES = [0.0, 10.0, 20.0, 30.0]


@pytest.fixture
def regional_grid():
    """Source rows at 0, 10 and 20N and columns at 0, 10, 20 and 30E."""
    return mask_cube(REGIONAL_LATITUDES, REGIONAL_LONGITUDES)


def regrid_onto(source_grid, source_mask, source_field, target_latitudes, target_longitudes):
    """Regrid onto an all-true target grid with the given axes."""
    regridder = build_regridder(
        with_mask(source_grid, source_mask), mask_cube(target_latitudes, target_longitudes), um_bilinear_weights
    )
    return regrid_values(regridder, source_field)


def test_corner_order_and_weights(regional_grid):
    """`bilinear_corners` returns the four source points around a target in the order SW, SE, NW,
    NE, with bilinear weights, here for a target 25% of the way north and 75% of the way east across
    its cell.
    """
    rows, columns, weights = bilinear_corners(regional_grid, [12.5], [17.5])
    np.testing.assert_array_equal(rows[:, 0], [1, 1, 2, 2])
    np.testing.assert_array_equal(columns[:, 0], [1, 2, 1, 2])
    # 25% of the way north from row 1 and 75% of the way east from column 1.
    np.testing.assert_allclose(weights[:, 0], [0.75 * 0.25, 0.75 * 0.75, 0.25 * 0.25, 0.25 * 0.75])


def test_interior_points_reproduce_linear_fields_exactly(regional_grid):
    """Where all four corners are valid, the value is bilinearly interpolated, which reproduces a
    field linear in latitude and longitude exactly at random target points.
    """
    field = 3.0 * true_latitudes(regional_grid) - 2.0 * true_longitudes(regional_grid) + 7.0
    mask = np.ones(regional_grid.shape, dtype=bool)
    random_generator = np.random.default_rng(0)
    target_latitudes = np.sort(random_generator.uniform(0.0, 20.0, 15))
    target_longitudes = np.sort(random_generator.uniform(0.0, 30.0, 14))
    regridded = regrid_onto(regional_grid, mask, field, target_latitudes, target_longitudes)
    expected = 3.0 * target_latitudes[:, None] - 2.0 * target_longitudes[None, :] + 7.0
    np.testing.assert_allclose(regridded, expected, atol=1e-12)


def test_coastal_point_copies_its_heaviest_valid_corner(regional_grid):
    """Where some corners are invalid, the point copies its valid corner with the largest weight.
    When two valid corners tie, the first in SW, SE, NW, NE order wins.
    """
    field = np.arange(12.0).reshape(3, 4)  # values equal to the flattened index
    mask = np.ones(regional_grid.shape, dtype=bool)
    # Target at (12.5, 17.5): corners SW (1,1), SE (1,2), NW (2,1), NE (2,2) with
    # weights 0.1875, 0.5625, 0.0625, 0.1875. Remove the heaviest (SE).
    mask[1, 2] = False
    regridded = regrid_onto(regional_grid, mask, field, [12.5], [17.5])
    # SW and NE tie at 0.1875; the earlier corner in SW, SE, NW, NE order wins.
    np.testing.assert_array_equal(regridded, [[field[1, 1]]])

    mask[1, 1] = False
    np.testing.assert_array_equal(regrid_onto(regional_grid, mask, field, [12.5], [17.5]), [[field[2, 2]]])


def test_coastal_point_copies_rather_than_blends(regional_grid):
    """A coastal point with three valid corners copies a single value rather than blending the
    three.
    """
    field = np.arange(12.0).reshape(3, 4)
    mask = np.ones(regional_grid.shape, dtype=bool)
    mask[2, 2] = False  # NE invalid; the other three corners are valid
    regridded = regrid_onto(regional_grid, mask, field, [12.5], [17.5])
    assert regridded[0, 0] == field[1, 2]


def test_unresolved_point_copies_nearest_resolved_target_point():
    """A target point with no valid corners copies the value of the nearest target point that has
    some (here the coastal point beside it), rather than searching the source itself.
    """
    source_grid = mask_cube([0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0, 40.0])
    source_mask = np.zeros(source_grid.shape, dtype=bool)
    source_mask[:, :2] = True  # valid only west of 10E
    field = np.arange(20.0).reshape(4, 5)
    # Targets along 15N: 5E is interior, 15E is coastal, 35E has no valid corners.
    regridded = regrid_onto(source_grid, source_mask, field, [15.0], [5.0, 15.0, 35.0])[0]
    assert regridded[2] == regridded[1]
    assert regridded[1] == field[1, 1]  # coastal: heaviest valid corner, SW and NW tie -> SW


def test_unresolved_west_east_tie_prefers_the_west():
    """When an unresolved point is equally far from resolved points to its west and east, it copies
    the western one.
    """
    source_grid = mask_cube([-10.0, 0.0, 10.0, 20.0, 30.0], np.arange(0.0, 100.0, 10.0))
    source_mask = np.zeros(source_grid.shape, dtype=bool)
    source_mask[:, :2] = True
    source_mask[:, 8:] = True
    field = np.arange(50.0).reshape(5, 10)
    target_grid = mask_cube([5.0, 15.0, 25.0], [5.0, 45.0, 85.0])
    target_mask = np.zeros(target_grid.shape, dtype=bool)
    target_mask[1, :] = True  # (15N, 45E) is unresolved and 40 degrees from both 5E and 85E
    regridded = regrid_values(
        build_regridder(with_mask(source_grid, source_mask), with_mask(target_grid, target_mask)), field
    )
    assert regridded[1, 0] != regridded[1, 2]
    assert regridded[1, 1] == regridded[1, 0]


def test_unresolved_north_south_tie_prefers_the_equator():
    """When an unresolved point is equally far from resolved points to its north and south, it
    copies the one nearer the equator.
    """
    # Source valid only in two bands; the target point between them has no valid corner.
    source_grid = mask_cube([-30.0, -20.0, -10.0, 0.0, 10.0], [0.0, 10.0, 20.0])
    source_mask = np.zeros(source_grid.shape, dtype=bool)
    source_mask[0, :] = True  # -30
    source_mask[4, :] = True  # +10
    field = np.where(np.arange(5)[:, None] == 0, 1.0, 2.0) * np.ones(source_grid.shape)
    target_grid = mask_cube([-25.0, -10.0, 5.0], [5.0, 6.0])
    target_mask = np.zeros(target_grid.shape, dtype=bool)
    target_mask[:, 0] = True
    regridded = regrid_values(
        build_regridder(with_mask(source_grid, source_mask), with_mask(target_grid, target_mask)), field
    )
    # -25 is resolved from the -30 band and 5 from the 10 band; -10 is 15 degrees
    # from both and takes the one nearer the equator.
    assert regridded[0, 0] == 1.0
    assert regridded[2, 0] == 2.0
    assert regridded[1, 0] == 2.0


def test_cyclic_grids_interpolate_across_the_wrap():
    """On a global grid, a target between the last and first columns interpolates across the 0/360
    wrap, whichever longitude convention it uses.
    """
    source_grid = global_mask_cube(30.0)  # columns at 15, 45, ..., 345
    field = np.cos(np.radians(true_longitudes(source_grid)))
    mask = np.ones(source_grid.shape, dtype=bool)
    # 0E lies midway between the 345E and 15E columns.
    expected = 0.5 * (np.cos(np.radians(345.0)) + np.cos(np.radians(15.0)))
    for target_longitude in (0.0, 360.0):
        regridded = regrid_onto(source_grid, mask, field, [0.0], [target_longitude])
        np.testing.assert_allclose(regridded, [[expected]], atol=1e-12)
    regridded = regrid_onto(source_grid, mask, field, [0.0], [-15.0])
    np.testing.assert_allclose(regridded, [[np.cos(np.radians(345.0))]], atol=1e-12)


def test_entries_are_stored_in_corner_order_across_the_wrap():
    """Across the wrap, weights stay stored in SW, SE, NW, NE order even though the SW corner has
    the higher column index. Values are then summed in the UM's order, keeping results bitwise
    identical.
    """
    source_grid = global_mask_cube(30.0)
    target_grid = mask_cube([0.0], [0.0])
    weights = um_bilinear_weights(source_grid, target_grid)
    # The SW corner (last column) has a higher column number than SE (first column),
    # but entries stay in SW, SE, NW, NE order so sums match the UM's bitwise.
    rows, columns, _ = bilinear_corners(source_grid, [0.0], [0.0])
    np.testing.assert_array_equal(weights.indices, np.ravel_multi_index((rows[:, 0], columns[:, 0]), source_grid.shape))


def test_points_beyond_the_edge_rows_take_the_edge_row_values():
    """Targets poleward of the outermost source rows (as N512's polar rows are of N96's) take the
    edge row's values, still interpolated along the row.
    """
    source_grid = global_mask_cube(10.0)  # rows at -85 ... 85
    field = true_latitudes(source_grid) + 0.01 * true_longitudes(source_grid)
    mask = np.ones(source_grid.shape, dtype=bool)
    regridded = regrid_onto(source_grid, mask, field, [-89.0, 89.0], [10.0])
    np.testing.assert_allclose(regridded[:, 0], [-85.0 + 0.1, 85.0 + 0.1], atol=1e-12)


def test_regional_grids_clamp_beyond_their_edges(regional_grid):
    """On a regional (non-wrapping) source grid, targets beyond the first or last column take the
    edge column's values.
    """
    field = np.arange(12.0).reshape(3, 4)
    mask = np.ones(regional_grid.shape, dtype=bool)
    regridded = regrid_onto(regional_grid, mask, field, [0.0], [-5.0, 40.0])
    np.testing.assert_array_equal(regridded, [[field[0, 0], field[0, 3]]])


def test_rotated_source_grid_is_interpolated_in_its_own_frame():
    """On a rotated source grid, interpolation happens in the grid's own rotated frame. A field
    equal to rotated latitude is reproduced exactly at unrotated target points, and a target on the
    same rotated pole at a subset of the source points gets the source values back.
    """
    north_pole = (37.5, 177.5)
    source_grid = regular_mask_cube(-5.0, -5.0, 1.0, 1.0, 11, 11, north_pole=north_pole)
    rotated_latitude_axis = horizontal_coordinates(source_grid)[0].points
    rotated_latitude_field = rotated_latitude_axis[:, None] * np.ones((1, 11))
    mask = np.ones(source_grid.shape, dtype=bool)

    # An unrotated target over the British Isles: rotated latitude is linear in the
    # source frame, so interpolation recovers it exactly.
    target_latitudes = [50.0, 51.5, 53.0, 55.0]
    target_longitudes = [-6.0, -3.0, 0.0, 1.5]
    regridded = regrid_onto(source_grid, mask, rotated_latitude_field, target_latitudes, target_longitudes)
    target_latitude_grid, target_longitude_grid = np.meshgrid(target_latitudes, target_longitudes, indexing="ij")
    expected, _ = rotate_pole(target_latitude_grid, target_longitude_grid, *north_pole)
    np.testing.assert_allclose(regridded, expected, atol=1e-9)

    # A target on the same rotated pole, at a subset of the source points, gets exact values back.
    target_grid = mask_cube(rotated_latitude_axis[::3], rotated_latitude_axis[::3], north_pole=north_pole)
    regridded = regrid_values(build_regridder(source_grid, target_grid), rotated_latitude_field)
    np.testing.assert_allclose(regridded, rotated_latitude_field[::3, ::3], atol=1e-9)


def test_no_resolved_targets_falls_back_to_nearest_source(regional_grid):
    """If no target point has any valid corner, each takes its nearest valid source point, a
    fallback the UM never meets for land fields.
    """
    field = np.arange(12.0).reshape(3, 4)
    mask = np.zeros(regional_grid.shape, dtype=bool)
    mask[0, 0] = True
    # Every target is far from the single valid source point, so none is resolved.
    regridded = regrid_onto(regional_grid, mask, field, [19.0, 20.0], [29.0, 30.0])
    np.testing.assert_array_equal(regridded, 0.0)


def test_rows_sum_to_one_on_a_masked_global_regrid():
    """On a global regrid with random masks, the matrix has a row per valid target and a column per
    valid source, rows sum to 1, and every stored weight is positive.
    """
    source_grid = global_mask_cube(5.0)
    target_grid = global_mask_cube(2.0)
    random_generator = np.random.default_rng(4)
    source_mask = random_generator.random(source_grid.shape) < 0.3
    target_mask = random_generator.random(target_grid.shape) < 0.5
    weights = um_bilinear_weights(with_mask(source_grid, source_mask), with_mask(target_grid, target_mask))
    assert weights.shape == (np.count_nonzero(target_mask), np.count_nonzero(source_mask))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert weights.data.min() > 0.0
