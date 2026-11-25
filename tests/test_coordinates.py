"""Tests of reading masks and grid coordinates from cubes, and of the pole rotation for rotated
grids.

Every regridding method depends on these: they decide which points are valid and where each point
actually is on the sphere.
"""

import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import pytest
from conftest import mask_cube, regular_mask_cube

from um_masked_regrid import (
    horizontal_coordinates,
    longitude_is_cyclic,
    mask_from_cube,
    rotate_pole,
    rotated_north_pole_degrees,
    true_point_coordinates,
    unit_vectors,
    unrotate_pole,
)


def angular_difference_degrees(first_longitudes, second_longitudes):
    """Smallest absolute difference between longitudes, allowing for the 360 degree wrap."""
    return np.abs((np.asarray(first_longitudes) - np.asarray(second_longitudes) + 180.0) % 360.0 - 180.0)


def test_unrotate_pole_identity_for_pole_at_180():
    """A pole at (90N, 180E) is the rotated-pole description of an unrotated grid, so unrotating
    must leave the coordinates unchanged.
    """
    rotated_latitudes = np.array([-60.0, 0.0, 45.0])
    rotated_longitudes = np.array([10.0, 170.0, 300.0])
    true_latitudes, true_longitudes = unrotate_pole(rotated_latitudes, rotated_longitudes, 90.0, 180.0)
    np.testing.assert_allclose(true_latitudes, rotated_latitudes, atol=1e-12)
    np.testing.assert_allclose(angular_difference_degrees(true_longitudes, rotated_longitudes), 0.0, atol=1e-12)


def test_unrotate_pole_known_point():
    """Checks one known answer: on the standard UK rotated grid (pole at 37.5N, 177.5E), rotated (0,
    0) is at true 52.5N, 2.5W.
    """
    # The rotated equator/meridian origin of the classic UK grid sits at 52.5N, 2.5W.
    true_latitude, true_longitude = unrotate_pole(0.0, 0.0, 37.5, 177.5)
    assert true_latitude == pytest.approx(52.5)
    assert true_longitude == pytest.approx(357.5)


def test_unrotate_pole_maps_rotated_pole_to_declared_pole_position():
    """The rotated grid's own north pole (rotated latitude 90) must land at the true position the
    coordinate system declares for it.
    """
    true_latitude, _ = unrotate_pole(90.0, 0.0, 37.5, 177.5)
    assert true_latitude == pytest.approx(37.5)
    # Just short of the pole the longitude is well defined too.
    true_latitude, true_longitude = unrotate_pole(89.999, 180.0, 37.5, 177.5)
    assert true_latitude == pytest.approx(37.5, abs=2e-3)
    assert angular_difference_degrees(true_longitude, 177.5) < 2e-3


@pytest.mark.parametrize(
    "north_pole_latitude_degrees, north_pole_longitude_degrees",
    [(37.5, 177.5), (60.0, -30.0), (-20.0, 100.0), (88.0, 0.0)],
)
def test_unrotate_pole_matches_cartopy(north_pole_latitude_degrees, north_pole_longitude_degrees):
    """`unrotate_pole` agrees with cartopy's RotatedPole transform at 500 random points, for poles
    in both hemispheres and one near the true pole. Skipped if cartopy is not installed.
    """
    cartopy_crs = pytest.importorskip("cartopy.crs")
    random_generator = np.random.default_rng(1)
    rotated_latitudes = random_generator.uniform(-85.0, 85.0, 500)
    rotated_longitudes = random_generator.uniform(-180.0, 180.0, 500)

    true_latitudes, true_longitudes = unrotate_pole(
        rotated_latitudes, rotated_longitudes, north_pole_latitude_degrees, north_pole_longitude_degrees
    )

    rotated_crs = cartopy_crs.RotatedPole(
        pole_longitude=north_pole_longitude_degrees, pole_latitude=north_pole_latitude_degrees
    )
    expected = cartopy_crs.PlateCarree().transform_points(rotated_crs, rotated_longitudes, rotated_latitudes)
    np.testing.assert_allclose(true_latitudes, expected[:, 1], atol=1e-8)
    # Compare longitudes as positions on the sphere so the tolerance is meaningful near the poles.
    np.testing.assert_allclose(
        unit_vectors(true_latitudes, true_longitudes),
        unit_vectors(expected[:, 1], expected[:, 0]),
        atol=1e-9,
    )


def test_unit_vectors_have_unit_length_and_expected_axes():
    """Unit vectors have length 1, and the points at 0E and 90E on the equator and at the poles map
    to the x, y and z axes. The nearest-point searches rely on this.
    """
    vectors = unit_vectors([0.0, 0.0, 90.0, -90.0], [0.0, 90.0, 123.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    np.testing.assert_allclose(vectors[0], [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(vectors[1], [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(vectors[2], [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(vectors[3], [0.0, 0.0, -1.0], atol=1e-15)


def test_unit_vectors_ignore_longitude_convention():
    """The same point written with 0 to 360 or -180 to 180 longitudes gives the same unit vector, so
    searches don't depend on the convention.
    """
    np.testing.assert_allclose(
        unit_vectors([10.0, -40.0], [350.0, 181.0]),
        unit_vectors([10.0, -40.0], [-10.0, -179.0]),
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "north_pole_latitude_degrees, north_pole_longitude_degrees",
    [(37.5, 177.5), (60.0, -30.0), (-20.0, 100.0), (90.0, 180.0), (90.0, 0.0)],
)
def test_rotate_pole_inverts_unrotate_pole(north_pole_latitude_degrees, north_pole_longitude_degrees):
    """Rotating with `rotate_pole` undoes `unrotate_pole` (a round trip back to the rotated
    coordinates) for several poles, including unrotated ones, and keeps the 2D array shape.
    """
    random_generator = np.random.default_rng(2)
    rotated_latitudes = random_generator.uniform(-89.0, 89.0, (20, 25))
    rotated_longitudes = random_generator.uniform(0.0, 360.0, (20, 25))
    true_latitudes, true_longitudes = unrotate_pole(
        rotated_latitudes, rotated_longitudes, north_pole_latitude_degrees, north_pole_longitude_degrees
    )
    recovered_latitudes, recovered_longitudes = rotate_pole(
        true_latitudes, true_longitudes, north_pole_latitude_degrees, north_pole_longitude_degrees
    )
    assert recovered_latitudes.shape == (20, 25)
    np.testing.assert_allclose(recovered_latitudes, rotated_latitudes, atol=1e-9)
    np.testing.assert_allclose(angular_difference_degrees(recovered_longitudes, rotated_longitudes), 0.0, atol=1e-9)


def test_rotate_pole_matches_cartopy():
    """`rotate_pole` (true to rotated coordinates) agrees with cartopy's RotatedPole transform at
    300 random points. Skipped if cartopy is not installed.
    """
    cartopy_crs = pytest.importorskip("cartopy.crs")
    random_generator = np.random.default_rng(3)
    true_latitudes = random_generator.uniform(-85.0, 85.0, 300)
    true_longitudes = random_generator.uniform(-180.0, 180.0, 300)
    rotated_latitudes, rotated_longitudes = rotate_pole(true_latitudes, true_longitudes, 37.5, 177.5)
    expected = cartopy_crs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5).transform_points(
        cartopy_crs.PlateCarree(), true_longitudes, true_latitudes
    )
    np.testing.assert_allclose(
        unit_vectors(rotated_latitudes, rotated_longitudes), unit_vectors(expected[:, 1], expected[:, 0]), atol=1e-9
    )


def test_unit_vectors_flatten_in_c_order():
    """Points of a 2D grid are numbered in C order (row by row), the order the weight matrices use:
    point 4 of a 2x3 grid is row 1, column 1.
    """
    latitudes, longitudes = true_point_coordinates(mask_cube([0.0, 10.0], [0.0, 20.0, 40.0]))
    vectors = unit_vectors(latitudes, longitudes)
    assert vectors.shape == (6, 3)
    np.testing.assert_allclose(vectors[4], unit_vectors([10.0], [20.0])[0])


def test_true_point_coordinates_shape_and_ordering():
    """`true_point_coordinates` returns 2D arrays built from the cube's axes, with latitude varying
    down the rows and longitude along the columns.
    """
    latitudes, longitudes = true_point_coordinates(mask_cube([-10.0, 0.0, 30.0], [100.0, 101.0, 105.0, 120.0]))
    assert latitudes.shape == (3, 4)
    np.testing.assert_array_equal(latitudes[:, 0], [-10.0, 0.0, 30.0])
    np.testing.assert_array_equal(longitudes[0, :], [100.0, 101.0, 105.0, 120.0])


def test_true_point_coordinates_of_a_rotated_grid():
    """On a rotated grid the cube's axes stay in rotated coordinates, but `true_point_coordinates`
    gives true positions: the grid's centre, rotated (0, 0), is at 52.5N, 357.5E.
    """
    cube = regular_mask_cube(-1.0, -1.0, 1.0, 1.0, 3, 3, north_pole=(37.5, 177.5))
    latitude_coord, _ = horizontal_coordinates(cube)
    # The axes stay in the rotated frame; the points are converted to true coordinates.
    np.testing.assert_array_equal(latitude_coord.points, [-1.0, 0.0, 1.0])
    latitudes, longitudes = true_point_coordinates(cube)
    assert latitudes[1, 1] == pytest.approx(52.5)
    assert longitudes[1, 1] == pytest.approx(357.5)


def test_rotated_north_pole():
    """`rotated_north_pole_degrees` returns None for an unrotated grid and the declared (latitude,
    longitude) of the pole for a rotated one.
    """
    assert rotated_north_pole_degrees(mask_cube([0.0, 1.0], [0.0, 1.0])) is None
    assert rotated_north_pole_degrees(mask_cube([0.0, 1.0], [0.0, 1.0], north_pole=(37.5, 177.5))) == (37.5, 177.5)


def test_rotated_grids_with_a_grid_longitude_offset_are_not_supported():
    """A rotated coordinate system with a non-zero `north_pole_grid_longitude` (an extra rotation
    about the pole) must raise rather than give wrong positions.
    """
    cube = mask_cube([0.0, 1.0], [0.0, 1.0], north_pole=(37.5, 177.5))
    coord_system = iris.coord_systems.RotatedGeogCS(37.5, 177.5, north_pole_grid_longitude=10.0)
    for coord in cube.dim_coords:
        coord.coord_system = coord_system
    with pytest.raises(NotImplementedError, match="north_pole_grid_longitude"):
        rotated_north_pole_degrees(cube)


@pytest.mark.parametrize(
    "longitudes_degrees, expected_is_cyclic",
    [
        (1.875 * np.arange(192) + 0.9375, True),  # N96 ENDGame
        (np.arange(0.0, 360.0, 30.0), True),
        (np.array([0.0, 10.0, 30.0, 330.0, 350.0]), True),  # variable spacing, gap of 10
        (np.arange(100.0, 160.0, 1.0), False),  # regional
        (np.arange(0.0, 300.0, 30.0), False),  # 90 degree gap
        (np.array([10.0]), False),
    ],
)
def test_cyclic_longitude_detection(longitudes_degrees, expected_is_cyclic):
    """Longitudes wrap around the globe when one more grid spacing closes 360 degrees, even if the
    coordinate isn't flagged circular. Regional grids, grids with a large gap and single columns
    don't wrap.
    """
    _, longitude_coord = horizontal_coordinates(mask_cube([-10.0, 10.0], longitudes_degrees))
    assert longitude_is_cyclic(longitude_coord) is expected_is_cyclic


def test_circular_longitude_coordinates_are_cyclic():
    """A longitude coordinate flagged `circular` wraps around the globe, even though its points
    cover only a small range.
    """
    _, longitude_coord = horizontal_coordinates(mask_cube([-10.0, 10.0], [100.0, 120.0], circular=True))
    assert longitude_is_cyclic(longitude_coord)


def test_coordinates_in_radians_are_converted_to_degrees():
    """Coordinates in radians are converted to degrees for use internally, without modifying the
    cube that was passed in.
    """
    cube = mask_cube([-10.0, 10.0], [20.0, 40.0])
    for coord in cube.dim_coords:
        coord.convert_units("radians")
    latitude_coord, longitude_coord = horizontal_coordinates(cube)
    np.testing.assert_allclose(latitude_coord.points, [-10.0, 10.0])
    np.testing.assert_allclose(longitude_coord.points, [20.0, 40.0])
    # The cube itself is left as it was.
    assert cube.coord("latitude").units == "radians"


@pytest.mark.parametrize(
    "latitudes_degrees, longitudes_degrees, message",
    [
        ([1.0, 0.0], [0.0, 1.0], "strictly increasing"),
        ([0.0, 1.0], [5.0, 4.0], "strictly increasing"),
        ([0.0, 1.0], [0.0, 360.0], "less than 360"),
    ],
)
def test_bad_axes_are_rejected(latitudes_degrees, longitudes_degrees, message):
    """Decreasing axes, and longitudes spanning 360 degrees or more (which would duplicate a
    column), are rejected with a clear error.
    """
    with pytest.raises(ValueError, match=message):
        horizontal_coordinates(mask_cube(latitudes_degrees, longitudes_degrees))


def test_latitude_and_longitude_must_be_the_last_two_dimensions_in_order():
    """A cube whose last two dimensions are longitude then latitude is rejected: the dimensions must
    be (..., latitude, longitude).
    """
    longitude_first_cube = mask_cube([0.0, 1.0, 2.0], [0.0, 1.0])
    longitude_first_cube.transpose()
    with pytest.raises(ValueError, match="latitude then longitude"):
        horizontal_coordinates(longitude_first_cube)


def test_horizontal_dimensions_need_dimension_coordinates():
    """The last two dimensions must have dimension coordinates, and a cube with fewer than two
    dimensions is rejected.
    """
    cube = iris.cube.Cube(np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="dimension coordinates"):
        horizontal_coordinates(cube)
    with pytest.raises(ValueError, match="last two"):
        horizontal_coordinates(iris.cube.Cube(np.ones(3, dtype=bool)))


def test_latitude_and_longitude_must_share_a_coordinate_system():
    """Latitude and longitude with different coordinate systems are rejected, since the point
    positions would be ambiguous.
    """
    cube = mask_cube([0.0, 1.0], [0.0, 1.0])
    cube.coord("longitude").coord_system = iris.coord_systems.GeogCS(6371000.0)
    with pytest.raises(ValueError, match="different coordinate systems"):
        horizontal_coordinates(cube)


def test_mask_cubes_must_hold_booleans():
    """Mask cubes must hold boolean data. Integer or fractional data (e.g. tile fractions) is
    rejected rather than guessed at, so callers decide the mask explicitly.
    """
    with pytest.raises(TypeError, match="boolean data"):
        mask_from_cube(mask_cube([0.0, 1.0], [0.0, 1.0], mask=np.ones((2, 2), dtype=int)))


def test_mask_cubes_must_be_two_dimensional():
    """Mask cubes must be 2D: a stack of masks (e.g. every tile at once) is rejected, while a 2D
    all-true mask cube is read as all true.
    """
    cube = mask_cube([0.0, 1.0], [0.0, 1.0])
    stacked = iris.cube.Cube(np.ones((3, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="must be 2D"):
        mask_from_cube(stacked)
    np.testing.assert_array_equal(mask_from_cube(cube), True)


def test_masked_entries_of_a_mask_cube_count_as_false():
    """Masked entries of a mask cube count as false whatever value lies beneath them. This is what
    iris gives for sea points of a land-packed field, e.g. `tile_fraction_cube.data > 0`.
    """
    mask = np.ma.masked_array([[True, True], [True, False]], mask=[[False, True], [False, False]])
    np.testing.assert_array_equal(
        mask_from_cube(mask_cube([0.0, 1.0], [0.0, 1.0], mask=mask)), [[True, False], [True, False]]
    )
