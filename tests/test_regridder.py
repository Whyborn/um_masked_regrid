"""Tests of `build_regridder` and the `CubeRegridder` it returns.

They cover the values it produces, how it handles masks and missing data, and the output cube it
builds. Most run on global 5 degree ("coarse") and 2 degree ("fine") grids with random masks, and
many run once per weight builder, because these properties should hold whatever the method.
"""

import functools

import dask.array
import iris.aux_factory
import iris.coords
import numpy as np
import pytest
from conftest import (
    field_cube,
    global_mask_cube,
    mask_cube,
    random_mask,
    regrid_values,
    regular_mask_cube,
    smooth_field,
    true_latitudes,
    true_longitudes,
    with_mask,
)
from scipy.sparse import csr_array

from um_masked_regrid import (
    EmptySourceMaskError,
    build_regridder,
    inverse_distance_weights,
    mask_from_cube,
    nearest_neighbour_weights,
    nearest_source_distances,
    um_bilinear_weights,
)

WEIGHT_BUILDERS = [
    pytest.param(um_bilinear_weights, id="um_bilinear"),
    pytest.param(nearest_neighbour_weights, id="nearest"),
    pytest.param(inverse_distance_weights, id="inverse_distance"),
]

# Stands in for whatever junk a model leaves at points where a field is undefined.
INVALID_SENTINEL = 1.0e30


def build_masked_regridder(source_grid, source_mask, target_grid, target_mask, weight_builder=um_bilinear_weights):
    """Build a regridder from two grids and a boolean mask for each."""
    return build_regridder(with_mask(source_grid, source_mask), with_mask(target_grid, target_mask), weight_builder)


# Values: what is regridded and where.


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_same_grid_and_mask_reproduces_the_field_exactly(weight_builder, coarse_grid):
    """Regridding onto the same grid with the same mask returns the field bitwise unchanged at every
    valid point, and the fill value (NaN) everywhere else.
    """
    mask = random_mask(coarse_grid, valid_fraction=0.4, seed=0)
    field = smooth_field(coarse_grid)
    regridder = build_masked_regridder(coarse_grid, mask, coarse_grid, mask, weight_builder)
    regridded = regrid_values(regridder, field)
    np.testing.assert_array_equal(regridded[mask], field[mask])
    assert np.all(np.isnan(regridded[~mask]))


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
@pytest.mark.parametrize("direction", ["refine", "coarsen"])
def test_masked_out_source_points_never_contribute(weight_builder, direction, coarse_grid, fine_grid):
    """Source points outside the source mask hold a huge sentinel value. Every regridded value stays
    within the range of the valid source values, so the sentinel never leaks in, and points outside
    the target mask get the fill value. Checked going from coarse to fine and from fine to coarse.
    """
    source_grid, target_grid = (coarse_grid, fine_grid) if direction == "refine" else (fine_grid, coarse_grid)
    source_mask = random_mask(source_grid, valid_fraction=0.3, seed=1)
    target_mask = random_mask(target_grid, valid_fraction=0.6, seed=2)
    source_field = np.where(source_mask, smooth_field(source_grid), INVALID_SENTINEL)

    regridder = build_masked_regridder(source_grid, source_mask, target_grid, target_mask, weight_builder)
    regridded = regrid_values(regridder, source_field, fill_value=-999.0)

    valid_source_values = source_field[source_mask]
    assert np.all(regridded[target_mask] >= valid_source_values.min())
    assert np.all(regridded[target_mask] <= valid_source_values.max())
    np.testing.assert_array_equal(regridded[~target_mask], -999.0)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_every_target_mask_point_is_filled_even_when_masks_barely_overlap(weight_builder, coarse_grid, fine_grid):
    """With the source defined only south of 10S and the target only north of 10N, every target
    point still gets a finite value, borrowed from far away.
    """
    # Source field exists only in the southern hemisphere, target only in the northern.
    source_mask = true_latitudes(coarse_grid) < -10.0
    target_mask = true_latitudes(fine_grid) > 10.0
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, weight_builder)
    regridded = regrid_values(regridder, smooth_field(coarse_grid))
    assert np.all(np.isfinite(regridded[target_mask]))
    assert regridder.weights.shape == (np.count_nonzero(target_mask), np.count_nonzero(source_mask))


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_constant_field_is_preserved(weight_builder, coarse_grid, fine_grid):
    """A constant field stays constant: each output is a weighted mean with weights summing to 1."""
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=3)
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=4)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, weight_builder)
    regridded = regrid_values(regridder, np.full(coarse_grid.shape, 273.15))
    np.testing.assert_allclose(regridded[target_mask], 273.15, rtol=0.0, atol=1e-10)


def test_nearest_neighbour_copies_source_values_unchanged(coarse_grid, fine_grid):
    """Nearest-neighbour regridding of integer categories gives only values present in the valid
    source, never blends, so it's safe for flags and categories.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=5)
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=6)
    # Integer categories: every output must be one of the inputs, never a blend.
    categories = np.random.default_rng(7).integers(1, 13, coarse_grid.shape)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, nearest_neighbour_weights)
    regridded = regrid_values(regridder, categories)
    assert np.all(np.isin(regridded[target_mask], categories[source_mask]))
    np.testing.assert_array_equal(regridded[target_mask], np.round(regridded[target_mask]))


def test_isolated_target_point_takes_the_only_available_source_value(coarse_grid, fine_grid):
    """When a surface type exists at only one source point, every target point of that type takes
    its value, whichever method is used.
    """
    # The surface type exists at one source point only, on the far side of the globe.
    source_mask = np.zeros(coarse_grid.shape, dtype=bool)
    source_mask[5, 7] = True
    source_field = np.full(coarse_grid.shape, INVALID_SENTINEL)
    source_field[5, 7] = 42.0
    target_mask = random_mask(fine_grid, valid_fraction=0.1, seed=8)
    for weight_builder in (um_bilinear_weights, nearest_neighbour_weights, inverse_distance_weights):
        regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, weight_builder)
        np.testing.assert_array_equal(regrid_values(regridder, source_field)[target_mask], 42.0)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_longitude_convention_does_not_change_the_result(weight_builder, coarse_grid):
    """Describing the same source grid with longitudes in -180 to 180 rather than 0 to 360 (with the
    data rolled to match) gives the same result, for every method.
    """
    # Offset the target so no target point is equidistant from two source points;
    # otherwise rounding differences between conventions could break ties differently.
    target_grid = regular_mask_cube(-89.23, 0.37, 2.0, 2.0, row_count=90, column_count=180)
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=9)
    target_mask = random_mask(target_grid, valid_fraction=0.5, seed=10)
    field = smooth_field(coarse_grid)

    # The same source grid with longitudes in [-180, 180): the eastern half moves to the front.
    longitudes_degrees = coarse_grid.coord("longitude").points
    half_column_count = longitudes_degrees.size // 2
    shifted_source_grid = mask_cube(
        coarse_grid.coord("latitude").points,
        np.concatenate((longitudes_degrees[half_column_count:] - 360.0, longitudes_degrees[:half_column_count])),
        circular=True,
    )

    def shift(values):
        return np.roll(values, -half_column_count, axis=-1)

    reference = regrid_values(
        build_masked_regridder(coarse_grid, source_mask, target_grid, target_mask, weight_builder), field
    )
    shifted = regrid_values(
        build_masked_regridder(shifted_source_grid, shift(source_mask), target_grid, target_mask, weight_builder),
        shift(field),
    )
    np.testing.assert_allclose(shifted, reference, rtol=0.0, atol=1e-9, equal_nan=True)


def test_inverse_distance_is_more_accurate_than_nearest_on_a_smooth_field(coarse_grid, fine_grid):
    """On a smooth field, inverse distance is closer to the exact values than nearest neighbour (at
    least 20% lower RMS error), and nearest neighbour's error is plausible for 5 degree spacing.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.7, seed=11)
    target_mask = random_mask(fine_grid, valid_fraction=0.7, seed=12)
    source_field = smooth_field(coarse_grid)
    exact = smooth_field(fine_grid)[target_mask]

    def root_mean_square_error(weight_builder):
        regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, weight_builder)
        return np.sqrt(np.mean((regrid_values(regridder, source_field)[target_mask] - exact) ** 2))

    nearest_error = root_mean_square_error(nearest_neighbour_weights)
    inverse_distance_error = root_mean_square_error(inverse_distance_weights)
    assert inverse_distance_error < 0.8 * nearest_error
    # Sanity bound: the field spans about 60 units and the source spacing is 5 degrees.
    assert nearest_error < 3.0


@pytest.fixture
def regional_rotated_grid():
    """A 41x41, 0.1 degree regional grid on the standard UK rotated pole, centred on rotated (0, 0),
    i.e. 52.5N, 2.5W.
    """
    # A 0.1 degree regional grid centred on rotated (0, 0), i.e. 52.5N, 2.5W.
    return regular_mask_cube(-2.0, -2.0, 0.1, 0.1, row_count=41, column_count=41, north_pole=(37.5, 177.5))


def test_regional_rotated_target_picks_neighbours_at_its_true_location(regional_rotated_grid):
    """Nearest-neighbour regridding of true latitude and longitude from a global 1 degree grid onto
    the rotated regional grid gives values within about half a source cell of each target point's
    true position. So rotated target points are located by their true positions, not their rotated
    coordinates.
    """
    source_grid = global_mask_cube(1.0)
    regridder = build_regridder(source_grid, regional_rotated_grid, nearest_neighbour_weights)

    regridded_latitudes = regrid_values(regridder, true_latitudes(source_grid))
    regridded_longitudes = regrid_values(regridder, true_longitudes(source_grid))
    # The nearest centre of a 1 degree grid is at most about half a cell away in each
    # direction (slightly more in latitude, because meridians converge).
    assert np.max(np.abs(regridded_latitudes - true_latitudes(regional_rotated_grid))) < 0.6
    longitude_errors = np.abs(
        (regridded_longitudes - true_longitudes(regional_rotated_grid) + 180.0) % 360.0 - 180.0
    )
    assert np.max(longitude_errors) < 0.6
    assert np.max(nearest_source_distances(regridder).data) < np.radians(0.6)
    assert regridded_latitudes[20, 20] == pytest.approx(52.5)


def test_regional_rotated_target_with_um_bilinear_reproduces_linear_fields(regional_rotated_grid):
    """With the default UM method, regridding true latitude onto the rotated regional grid
    reproduces each target point's true latitude exactly, since bilinear interpolation is exact for
    linear fields.
    """
    # Latitude is linear in latitude, so bilinear interpolation reproduces it exactly
    # wherever the stencil does not straddle a pole.
    source_grid = global_mask_cube(1.0)
    regridder = build_regridder(source_grid, regional_rotated_grid)
    regridded_latitudes = regrid_values(regridder, true_latitudes(source_grid))
    np.testing.assert_allclose(
        regridded_latitudes, true_latitudes(regional_rotated_grid), rtol=0.0, atol=1e-9
    )


def test_default_method_needs_two_source_points_along_each_axis():
    """The UM method needs at least two source points along each axis to find bilinear corners, so a
    single-row source grid is rejected.
    """
    source_grid = mask_cube([0.0], [0.0, 10.0])
    with pytest.raises(ValueError, match="at least two source points"):
        build_regridder(source_grid, source_grid)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_leading_dimensions_match_level_by_level_application(weight_builder, coarse_grid, fine_grid):
    """Regridding a cube with two leading dimensions in one call gives the same result as regridding
    each 2D slice separately.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=13)
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=14)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, target_mask, weight_builder)
    levels = np.random.default_rng(15).random((2, 3) + coarse_grid.shape)

    regridded = regrid_values(regridder, levels)

    assert regridded.shape == (2, 3) + fine_grid.shape
    for first_index in range(2):
        for second_index in range(3):
            np.testing.assert_allclose(
                regridded[first_index, second_index],
                regrid_values(regridder, levels[first_index, second_index]),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )


def test_weight_builder_options_are_bound_with_partial(coarse_grid, fine_grid):
    """Builder options are bound with `functools.partial`: inverse distance with `neighbour_count=6`
    stores at most 6 weights per row.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=16)
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=17)
    regridder = build_masked_regridder(
        coarse_grid,
        source_mask,
        fine_grid,
        target_mask,
        functools.partial(inverse_distance_weights, neighbour_count=6),
    )
    assert np.max(np.diff(regridder.weights.indptr)) == 6


# Masks and missing data.


def test_empty_source_mask_with_target_points_raises(coarse_grid, fine_grid):
    """An empty source mask with a non-empty target mask raises `EmptySourceMaskError` (a
    ValueError), since there are no values to regrid and what to do instead is the caller's
    decision.
    """
    empty_source_mask = np.zeros(coarse_grid.shape, dtype=bool)
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=18)
    with pytest.raises(EmptySourceMaskError, match="source mask is empty"):
        build_masked_regridder(coarse_grid, empty_source_mask, fine_grid, target_mask)
    assert issubclass(EmptySourceMaskError, ValueError)


@pytest.mark.parametrize("source_is_empty", [False, True])
def test_empty_target_mask_gives_an_all_fill_field(source_is_empty, coarse_grid, fine_grid):
    """An empty target mask is valid and gives an output that is entirely fill value, even when the
    source mask is empty too.
    """
    source_mask = (
        np.zeros(coarse_grid.shape, dtype=bool)
        if source_is_empty
        else random_mask(coarse_grid, valid_fraction=0.5, seed=19)
    )
    empty_target_mask = np.zeros(fine_grid.shape, dtype=bool)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, empty_target_mask)
    regridded = regrid_values(regridder, smooth_field(coarse_grid), fill_value=-1.0)
    assert regridded.shape == fine_grid.shape
    np.testing.assert_array_equal(regridded, -1.0)


def test_non_boolean_mask_cubes_are_rejected(coarse_grid, fine_grid):
    """Source or target mask cubes holding fractions or integers are rejected, so masks are always
    made explicitly (e.g. `tile_fraction > 0`).
    """
    tile_fractions = np.random.default_rng(20).random(coarse_grid.shape)
    with pytest.raises(TypeError, match="boolean data"):
        build_regridder(with_mask(coarse_grid, tile_fractions), fine_grid)
    with pytest.raises(TypeError, match="boolean data"):
        build_regridder(coarse_grid, with_mask(fine_grid, np.ones(fine_grid.shape, dtype=int)))


def test_missing_data_inside_the_source_mask_is_rejected(coarse_grid, fine_grid):
    """A NaN, or a value equal to `source_missing_value`, inside the source mask raises an error,
    because the mask then doesn't describe where the field is defined. The same values outside the
    mask are ignored.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=21)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, np.ones(fine_grid.shape, dtype=bool))
    first_valid_index = tuple(np.argwhere(source_mask)[0])
    um_missing_value = -1073741824.0

    field_with_missing_value = smooth_field(coarse_grid)
    field_with_missing_value[first_valid_index] = um_missing_value
    with pytest.raises(ValueError, match="missing value"):
        regrid_values(regridder, field_with_missing_value, source_missing_value=um_missing_value)

    field_with_nan = smooth_field(coarse_grid)
    field_with_nan[first_valid_index] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        regrid_values(regridder, field_with_nan)

    # Missing data outside the mask is expected and ignored.
    field_missing_outside_mask = np.where(source_mask, smooth_field(coarse_grid), um_missing_value)
    regridded = regrid_values(regridder, field_missing_outside_mask, source_missing_value=um_missing_value)
    assert regridded.min() > 200.0


def test_masked_data_is_allowed_only_outside_the_source_mask(coarse_grid, fine_grid):
    """Masked data outside the source mask gives the same result as unmasked data, while a masked
    value inside the source mask raises an error.
    """
    source_mask = random_mask(coarse_grid, valid_fraction=0.5, seed=22)
    regridder = build_masked_regridder(coarse_grid, source_mask, fine_grid, np.ones(fine_grid.shape, dtype=bool))
    field = smooth_field(coarse_grid)

    masked_outside = np.ma.masked_array(field, mask=~source_mask)
    np.testing.assert_array_equal(
        regridder(field_cube(coarse_grid, masked_outside)).data, regridder(field_cube(coarse_grid, field)).data
    )

    masked_inside = np.ma.masked_array(field, mask=~source_mask)
    masked_inside[tuple(np.argwhere(source_mask)[0])] = np.ma.masked
    with pytest.raises(ValueError, match="masked values inside the source mask"):
        regridder(field_cube(coarse_grid, masked_inside))


def test_regridder_keeps_its_own_copies_of_the_mask_cubes(coarse_grid, fine_grid):
    """The regridder copies the mask cubes it was built from, so changing the caller's cube
    afterwards doesn't change the regridder.
    """
    target_mask_cube = with_mask(fine_grid, random_mask(fine_grid, valid_fraction=0.5, seed=23))
    regridder = build_regridder(coarse_grid, target_mask_cube)
    expected_target_mask = target_mask_cube.data.copy()
    target_mask_cube.data[:] = False
    np.testing.assert_array_equal(mask_from_cube(regridder.target_mask_cube), expected_target_mask)


def _wrong_shape_builder(source_mask_cube, target_mask_cube):
    """A broken builder whose matrix has one column too many."""
    return csr_array(
        (np.count_nonzero(mask_from_cube(target_mask_cube)), np.count_nonzero(mask_from_cube(source_mask_cube)) + 1)
    )


def _unnormalised_builder(source_mask_cube, target_mask_cube):
    """A broken builder whose rows sum to 2."""
    return 2.0 * nearest_neighbour_weights(source_mask_cube, target_mask_cube)


def _negative_builder(source_mask_cube, target_mask_cube):
    """A broken builder whose rows sum to 1 but include a negative weight."""
    source_point_count = np.count_nonzero(mask_from_cube(source_mask_cube))
    row = [2.0, -1.0] + [0.0] * (source_point_count - 2)
    return csr_array(np.tile(row, (np.count_nonzero(mask_from_cube(target_mask_cube)), 1)))


@pytest.mark.parametrize(
    "bad_weight_builder, message",
    [
        (_wrong_shape_builder, "shape"),
        (_unnormalised_builder, "sum to 1"),
        (_negative_builder, "negative"),
    ],
)
def test_weight_builders_that_break_the_contract_are_rejected(bad_weight_builder, message):
    """`build_regridder` rejects builders that return a matrix of the wrong shape, rows that don't
    sum to 1, or negative weights.
    """
    grid = global_mask_cube(30.0)
    with pytest.raises(ValueError, match=message):
        build_regridder(grid, grid, bad_weight_builder)


# The output cube.


def test_data_cube_on_a_different_grid_is_rejected(coarse_grid, fine_grid):
    """Calling the regridder on a cube that isn't on the source mask cube's grid raises, for a
    different resolution and for longitudes shifted by only 0.1 degrees.
    """
    regridder = build_regridder(coarse_grid, fine_grid)
    with pytest.raises(ValueError, match="not on the source mask cube's grid"):
        regridder(field_cube(fine_grid, smooth_field(fine_grid)))
    shifted_grid = global_mask_cube(5.0)
    shifted_grid.coord("longitude").points = shifted_grid.coord("longitude").points + 0.1
    with pytest.raises(ValueError, match="'longitude' coordinate differs"):
        regridder(field_cube(shifted_grid, smooth_field(shifted_grid)))


def test_output_cube_has_the_source_metadata_and_the_target_grid(coarse_grid, fine_grid):
    """The output cube keeps the source cube's name, units, attributes, leading coordinates (depth,
    layer number) and scalar coordinates (time). Its latitude, longitude and 2D auxiliary
    coordinates (true latitude) come from the target mask cube. Coordinates on the old grid (surface
    altitude) and the mask cube's scalar coordinates (pseudo level) are dropped.
    """
    depth_coord = iris.coords.DimCoord([0.05, 0.2, 0.5], standard_name="depth", units="m")
    source_cube = field_cube(
        coarse_grid,
        np.stack([smooth_field(coarse_grid)] * 3),
        leading_coords=[(depth_coord, 0)],
    )
    source_cube.standard_name = "soil_temperature"
    source_cube.attributes["STASH"] = "m01s00i020"
    source_cube.add_aux_coord(iris.coords.AuxCoord(1.0, standard_name="time", units="days since 2000-01-01"))
    source_cube.add_aux_coord(iris.coords.AuxCoord([1, 2, 3], long_name="layer_number"), 0)
    # Describes the old grid, so must not survive.
    source_cube.add_aux_coord(
        iris.coords.AuxCoord(np.zeros(coarse_grid.shape), standard_name="surface_altitude", units="m"), (1, 2)
    )

    target_mask_cube = fine_grid.copy()
    target_mask_cube.add_aux_coord(
        iris.coords.AuxCoord(true_latitudes(fine_grid), long_name="true_latitude", units="degrees"), (0, 1)
    )
    target_mask_cube.add_aux_coord(iris.coords.AuxCoord(17, long_name="pseudo_level"))

    regridded = build_regridder(coarse_grid, target_mask_cube)(source_cube)

    assert regridded.shape == (3,) + fine_grid.shape
    assert regridded.standard_name == "soil_temperature"
    assert regridded.units == "K"
    assert regridded.attributes["STASH"] == "m01s00i020"
    assert regridded.coord("depth") == depth_coord
    assert regridded.coord_dims("depth") == (0,)
    assert regridded.coord("layer_number").points.tolist() == [1, 2, 3]
    assert regridded.coord("time").points[0] == 1.0
    assert regridded.coord("latitude") == fine_grid.coord("latitude")
    assert regridded.coord("longitude") == fine_grid.coord("longitude")
    assert regridded.coord_dims("latitude") == (1,)
    assert regridded.coord_dims("longitude") == (2,)
    assert regridded.coord_dims("true_latitude") == (1, 2)
    assert not regridded.coords("surface_altitude")
    # Scalar coordinates of the mask describe the mask, not the field.
    assert not regridded.coords("pseudo_level")


def test_aux_factories_survive_only_if_their_dependencies_do(coarse_grid, fine_grid):
    """A hybrid-height factory is kept if it depends only on leading coordinates, and dropped if it
    depends on orography on the old grid.
    """
    level_coord = iris.coords.DimCoord([0, 1], long_name="model_level_number")
    source_cube = field_cube(
        coarse_grid, np.stack([smooth_field(coarse_grid)] * 2), leading_coords=[(level_coord, 0)]
    )
    level_height = iris.coords.AuxCoord([10.0, 20.0], long_name="level_height", units="m")
    sigma = iris.coords.AuxCoord([0.9, 0.8], long_name="sigma")
    orography = iris.coords.AuxCoord(np.zeros(coarse_grid.shape), standard_name="surface_altitude", units="m")
    source_cube.add_aux_coord(level_height, 0)
    source_cube.add_aux_coord(sigma, 0)
    regridder = build_regridder(coarse_grid, fine_grid)

    without_orography = source_cube.copy()
    without_orography.add_aux_factory(
        iris.aux_factory.HybridHeightFactory(
            without_orography.coord("level_height"), without_orography.coord("sigma")
        )
    )
    assert len(regridder(without_orography).aux_factories) == 1

    with_orography = source_cube.copy()
    with_orography.add_aux_coord(orography, (1, 2))
    with_orography.add_aux_factory(
        iris.aux_factory.HybridHeightFactory(
            with_orography.coord("level_height"), with_orography.coord("sigma"), orography
        )
    )
    assert len(regridder(with_orography).aux_factories) == 0


def test_output_is_masked_outside_the_target_mask_unless_a_fill_value_is_given(coarse_grid, fine_grid):
    """By default the output is a float64 masked array, masked exactly outside the target mask. With
    `fill_value` it's a plain array holding `fill_value` there, and the same values inside the mask.
    """
    target_mask = random_mask(fine_grid, valid_fraction=0.5, seed=24)
    regridder = build_masked_regridder(coarse_grid, np.ones(coarse_grid.shape, dtype=bool), fine_grid, target_mask)
    source_cube = field_cube(coarse_grid, np.stack([smooth_field(coarse_grid)] * 2))

    masked_output = regridder(source_cube).data
    assert np.ma.isMaskedArray(masked_output)
    assert masked_output.dtype == np.float64
    np.testing.assert_array_equal(np.ma.getmaskarray(masked_output), np.broadcast_to(~target_mask, (2,) + fine_grid.shape))

    filled_output = regridder(source_cube, fill_value=-1.0).data
    assert not np.ma.isMaskedArray(filled_output)
    np.testing.assert_array_equal(filled_output[:, ~target_mask], -1.0)
    np.testing.assert_array_equal(filled_output[:, target_mask], masked_output[:, target_mask])


def test_lazy_data_cubes_are_regridded(coarse_grid, fine_grid):
    """A cube with lazy (dask) data gives the same result as one with real data."""
    regridder = build_regridder(coarse_grid, fine_grid)
    field = smooth_field(coarse_grid)
    lazy_cube = field_cube(coarse_grid, dask.array.from_array(field, chunks=(9, 18)))
    assert lazy_cube.has_lazy_data()
    np.testing.assert_array_equal(regridder(lazy_cube).data, regridder(field_cube(coarse_grid, field)).data)


def test_regridding_does_not_change_the_source_cube(coarse_grid, fine_grid):
    """Regridding leaves the source cube unchanged."""
    source_cube = field_cube(coarse_grid, smooth_field(coarse_grid))
    expected = source_cube.copy()
    build_regridder(coarse_grid, fine_grid)(source_cube)
    assert source_cube == expected


def test_nearest_source_distances(coarse_grid):
    """`nearest_source_distances` gives each valid target point's distance to the nearest valid
    source point, as a cube on the target grid in radians, masked outside the target mask. With only
    one source row valid, a target on that row is 0 away and one 4 rows (20 degrees) north is 20
    degrees away.
    """
    # Source valid only on one row; distance grows with latitude separation along a meridian.
    source_mask = np.zeros(coarse_grid.shape, dtype=bool)
    source_mask[18, :] = True
    target_mask = np.zeros(coarse_grid.shape, dtype=bool)
    target_mask[18, 3] = True
    target_mask[22, 3] = True
    regridder = build_masked_regridder(coarse_grid, source_mask, coarse_grid, target_mask)
    distances = nearest_source_distances(regridder)
    assert distances.shape == coarse_grid.shape
    assert distances.units == "radians"
    assert distances.coord("latitude") == coarse_grid.coord("latitude")
    assert distances.data[18, 3] == 0.0
    assert distances.data[22, 3] == pytest.approx(np.radians(20.0))
    np.testing.assert_array_equal(np.ma.getmaskarray(distances.data), ~target_mask)
