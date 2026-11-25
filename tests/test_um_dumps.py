"""Checks against real UM start dumps, using tile fractions as the masks.

These load the dumps in the test data directory (see conftest.py) with iris and are
skipped if they are absent.
"""

import numpy as np
import pytest
from conftest import TEST_DATA_DIRECTORY
from um_dump_loading import LAND_SEA_MASK_STASH_CODE, boolean_mask_cube, load_dump_cubes

from um_masked_regrid import (
    EmptySourceMaskError,
    build_regridder,
    inverse_distance_weights,
    mask_from_cube,
    nearest_neighbour_weights,
    nearest_source_distances,
    true_point_coordinates,
    um_bilinear_weights,
)

SOURCE_DUMP_PATH = TEST_DATA_DIRECTORY / "am3a.astart_n96e"
TARGET_DUMP_PATH = TEST_DATA_DIRECTORY / "am3a.astart_n48e"

pytestmark = pytest.mark.skipif(
    not (SOURCE_DUMP_PATH.exists() and TARGET_DUMP_PATH.exists()),
    reason=f"UM test dumps are not available in {TEST_DATA_DIRECTORY}",
)

TILE_FRACTION_STASH_CODE = 216
CABLE_SOIL_TEMPERATURE_LAYER_1_STASH_CODE = 801
STASH_CODES = (LAND_SEA_MASK_STASH_CODE, TILE_FRACTION_STASH_CODE, CABLE_SOIL_TEMPERATURE_LAYER_1_STASH_CODE)

WEIGHT_BUILDERS = [
    pytest.param(um_bilinear_weights, id="um_bilinear"),
    pytest.param(nearest_neighbour_weights, id="nearest"),
    pytest.param(inverse_distance_weights, id="inverse_distance"),
]


def tile_mask_cubes(dump_cubes):
    """One 2D mask cube per tile: true on land where the tile fraction is positive."""
    is_land = dump_cubes[LAND_SEA_MASK_STASH_CODE].data != 0
    tile_fraction_cube = dump_cubes[TILE_FRACTION_STASH_CODE]
    return [
        boolean_mask_cube(tile_fraction_slice, (np.ma.filled(tile_fraction_slice.data, 0.0) > 0.0) & is_land)
        for tile_fraction_slice in tile_fraction_cube.slices(["latitude", "longitude"])
    ]


@pytest.fixture(scope="module")
def source_cubes():
    return load_dump_cubes(SOURCE_DUMP_PATH, STASH_CODES)


@pytest.fixture(scope="module")
def target_cubes():
    return load_dump_cubes(TARGET_DUMP_PATH, STASH_CODES)


@pytest.fixture(scope="module")
def source_tile_masks(source_cubes):
    return tile_mask_cubes(source_cubes)


@pytest.fixture(scope="module")
def target_tile_masks(target_cubes):
    return tile_mask_cubes(target_cubes)


def test_dump_grids_are_global_and_cell_centred(source_cubes, target_cubes):
    for dump_cubes, spacing_degrees in ((source_cubes, 1.25), (target_cubes, 2.5)):
        latitudes_degrees, _ = true_point_coordinates(dump_cubes[LAND_SEA_MASK_STASH_CODE])
        assert latitudes_degrees[0, 0] == pytest.approx(-90.0 + spacing_degrees / 2.0)
        assert latitudes_degrees[-1, 0] == pytest.approx(90.0 - spacing_degrees / 2.0)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_every_tile_regrids_onto_the_target_tile_mask(
    weight_builder, source_cubes, source_tile_masks, target_tile_masks
):
    soil_temperature_tiles = list(
        source_cubes[CABLE_SOIL_TEMPERATURE_LAYER_1_STASH_CODE].slices(["latitude", "longitude"])
    )
    filled_tile_count = 0
    for source_mask_cube, target_mask_cube, soil_temperature_cube in zip(
        source_tile_masks, target_tile_masks, soil_temperature_tiles
    ):
        target_mask = mask_from_cube(target_mask_cube)
        if not target_mask.any():
            continue
        regridder = build_regridder(source_mask_cube, target_mask_cube, weight_builder)
        regridded = regridder(soil_temperature_cube).data

        valid_source_values = np.ma.getdata(soil_temperature_cube.data)[mask_from_cube(source_mask_cube)]
        assert regridded.shape == target_mask.shape
        assert np.all(regridded[target_mask] >= valid_source_values.min())
        assert np.all(regridded[target_mask] <= valid_source_values.max())
        # Physically sensible soil temperatures, i.e. no zeros leaked in from inactive tiles.
        assert regridded[target_mask].min() > 180.0
        np.testing.assert_array_equal(np.ma.getmaskarray(regridded), ~target_mask)
        filled_tile_count += 1
    assert filled_tile_count >= 10


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_regridding_a_dump_onto_itself_is_exact(weight_builder, source_cubes, source_tile_masks):
    soil_temperature_tiles = source_cubes[CABLE_SOIL_TEMPERATURE_LAYER_1_STASH_CODE].slices(["latitude", "longitude"])
    for tile_mask_cube, soil_temperature_cube in zip(source_tile_masks, soil_temperature_tiles):
        tile_mask = mask_from_cube(tile_mask_cube)
        if not tile_mask.any():
            continue
        regridded = build_regridder(tile_mask_cube, tile_mask_cube, weight_builder)(soil_temperature_cube)
        np.testing.assert_array_equal(
            regridded.data[tile_mask], np.ma.getdata(soil_temperature_cube.data)[tile_mask]
        )


def test_tile_missing_from_the_source_raises(source_tile_masks, target_tile_masks):
    # Go from the N48 dump to the N96 one: at least one tile exists only in the latter.
    unfillable_tile_indices = [
        tile_index
        for tile_index, (n96_mask_cube, n48_mask_cube) in enumerate(zip(source_tile_masks, target_tile_masks))
        if mask_from_cube(n96_mask_cube).any() and not mask_from_cube(n48_mask_cube).any()
    ]
    if not unfillable_tile_indices:
        pytest.skip("every tile in the N96 dump also exists in the N48 dump")
    for tile_index in unfillable_tile_indices:
        with pytest.raises(EmptySourceMaskError):
            build_regridder(target_tile_masks[tile_index], source_tile_masks[tile_index])


def test_borrow_distances_are_reported_on_the_target_mask(source_tile_masks, target_tile_masks):
    regridder = build_regridder(source_tile_masks[0], target_tile_masks[0])
    distances = nearest_source_distances(regridder)
    target_mask = mask_from_cube(target_tile_masks[0])
    assert np.all(np.isfinite(distances.data[target_mask]))
    np.testing.assert_array_equal(np.ma.getmaskarray(distances.data), ~target_mask)
    assert distances.data[target_mask].max() <= np.pi
