"""Compare `um_bilinear_weights` with the UM's own reconfiguration of real dumps.

The source is the N96 dump the native reconfigurations started from. Targets
are its native reconfiguration to N48 (grid points aligned with the source) and
to N512 (not aligned). The land-sea mask is the mask throughout.

Fields with no post-processing in the reconfiguration must agree bitwise, apart
from N512 coastal points whose heaviest valid corners tie exactly in weight;
there the UM's choice is not reproducible (see the README). Soil temperature
and soil moisture are post-processed by the UM, so these tests apply the same
adjustments inline.

Everything goes through the public cube interface: fields are loaded with
`iris.load` and regridded by calling the regridder on them.

Skipped if the dumps are not available.
"""

import numpy as np
import pytest
from conftest import TEST_DATA_DIRECTORY
from um_dump_loading import (
    LAND_SEA_MASK_STASH_CODE,
    boolean_mask_cube,
    land_mask_cube,
    layer_thicknesses_from_mid_depths,
    load_dump_cubes,
)

from um_masked_regrid import (
    bilinear_corners,
    build_regridder,
    mask_from_cube,
    true_point_coordinates,
)

SOURCE_DUMP_PATH = TEST_DATA_DIRECTORY / "am3a.ainitial_n96e"
TARGET_DUMP_PATHS = {
    "n48e": TEST_DATA_DIRECTORY / "am3a.astart_n48e",
    "n512e": TEST_DATA_DIRECTORY / "am3a.astart_n512e",
}

CANOPY_WATER_STASH_CODE = 22
CANOPY_CONDUCTANCE_STASH_CODE = 213
TILE_CANOPY_WATER_STASH_CODE = 229
SOIL_TEMPERATURE_STASH_CODE = 20
SOIL_MOISTURE_STASH_CODE = 9
OROGRAPHY_STASH_CODE = 33
WILTING_POINT_STASH_CODE = 40
CRITICAL_POINT_STASH_CODE = 41
UNPROCESSED_STASH_CODES = (
    CANOPY_WATER_STASH_CODE,
    CANOPY_CONDUCTANCE_STASH_CODE,
    TILE_CANOPY_WATER_STASH_CODE,
)
ALL_STASH_CODES = UNPROCESSED_STASH_CODES + (
    LAND_SEA_MASK_STASH_CODE,
    SOIL_TEMPERATURE_STASH_CODE,
    SOIL_MOISTURE_STASH_CODE,
    OROGRAPHY_STASH_CODE,
    WILTING_POINT_STASH_CODE,
    CRITICAL_POINT_STASH_CODE,
)

# The reconfiguration lowers soil temperatures by this much per metre of extra
# orography (confirmed against both native dumps).
SOIL_TEMPERATURE_LAPSE_RATE_KELVIN_PER_METRE = 0.006
WATER_DENSITY_KILOGRAMS_PER_CUBIC_METRE = 1000.0

# The UM is built to flush denormal numbers to zero; numpy keeps them.
DENORMAL_THRESHOLD = 1.0e-300

# Fraction of land points per soil layer where the inline soil-moisture stress
# round trip reproduces the UM. The remainder are mostly next to source points
# whose wilting and critical points are equal (land ice), where the UM's
# treatment has not been worked out.
MINIMUM_SOIL_MOISTURE_AGREEMENT = 0.94


def _available_target_names():
    return [name for name, path in TARGET_DUMP_PATHS.items() if path.exists()]


pytestmark = pytest.mark.skipif(
    not SOURCE_DUMP_PATH.exists() or not _available_target_names(),
    reason=f"native reconfiguration test dumps are not available in {TEST_DATA_DIRECTORY}",
)


@pytest.fixture(scope="module")
def source_cubes():
    return load_dump_cubes(SOURCE_DUMP_PATH, ALL_STASH_CODES)


@pytest.fixture(scope="module")
def source_land_mask_cube(source_cubes):
    return land_mask_cube(source_cubes[LAND_SEA_MASK_STASH_CODE])


@pytest.fixture(scope="module", params=sorted(TARGET_DUMP_PATHS))
def target_name(request):
    if not TARGET_DUMP_PATHS[request.param].exists():
        pytest.skip(f"{TARGET_DUMP_PATHS[request.param]} is not available")
    return request.param


@pytest.fixture(scope="module")
def target_cubes(target_name):
    return load_dump_cubes(TARGET_DUMP_PATHS[target_name], ALL_STASH_CODES)


@pytest.fixture(scope="module")
def target_land_mask_cube(target_cubes):
    return land_mask_cube(target_cubes[LAND_SEA_MASK_STASH_CODE])


@pytest.fixture(scope="module")
def target_land_mask(target_land_mask_cube):
    return mask_from_cube(target_land_mask_cube)


@pytest.fixture(scope="module")
def land_regridder(source_land_mask_cube, target_land_mask_cube):
    return build_regridder(source_land_mask_cube, target_land_mask_cube)


@pytest.fixture(scope="module")
def coastal_weight_tie_points(source_land_mask_cube, target_land_mask_cube, target_land_mask):
    """Target land points that are coastal and whose heaviest valid corners tie exactly."""
    corner_rows, corner_columns, corner_weights = bilinear_corners(
        source_land_mask_cube, *true_point_coordinates(target_land_mask_cube)
    )
    corner_is_valid = mask_from_cube(source_land_mask_cube)[corner_rows, corner_columns]
    valid_corner_counts = corner_is_valid.sum(axis=0).reshape(target_land_mask.shape)
    valid_corner_weights = np.where(corner_is_valid, corner_weights, -1.0)
    heaviest_is_tied = (
        (valid_corner_weights == valid_corner_weights.max(axis=0)).sum(axis=0) > 1
    ).reshape(target_land_mask.shape)
    is_coastal = (valid_corner_counts > 0) & (valid_corner_counts < 4)
    return target_land_mask & is_coastal & heaviest_is_tied


@pytest.fixture(scope="module")
def points_depending_on_ties(land_regridder, coastal_weight_tie_points, target_land_mask):
    """Tie points, plus unresolved points that copy their value from a tie point."""
    tie_row_numbers = coastal_weight_tie_points[target_land_mask]
    tie_rows = land_regridder.weights[np.flatnonzero(tie_row_numbers)]
    tie_row_signatures = {
        (tuple(tie_rows.indices[start:end]), tuple(tie_rows.data[start:end]))
        for start, end in zip(tie_rows.indptr[:-1], tie_rows.indptr[1:])
    }
    weights = land_regridder.weights
    depends_on_tie = np.array(
        [
            (tuple(weights.indices[start:end]), tuple(weights.data[start:end])) in tie_row_signatures
            for start, end in zip(weights.indptr[:-1], weights.indptr[1:])
        ]
    )
    on_target_grid = np.zeros(target_land_mask.shape, dtype=bool)
    on_target_grid[target_land_mask] = depends_on_tie
    return on_target_grid | coastal_weight_tie_points


def _flush_denormals(values):
    return np.where(np.abs(values) < DENORMAL_THRESHOLD, 0.0, values)


def _any_over_leading_dimensions(differs, grid_shape):
    """True at grid points that differ on any level or pseudo-level."""
    return differs.reshape((-1,) + grid_shape).any(axis=0)


def _values(cube):
    """A cube's data as a plain float64 array, with masked (sea) points set to zero."""
    return np.ma.filled(cube.data.astype(np.float64), 0.0)


def test_regridded_cubes_carry_the_native_target_coordinates(
    source_cubes, target_cubes, land_regridder, target_land_mask
):
    for stash_code in (TILE_CANOPY_WATER_STASH_CODE, SOIL_MOISTURE_STASH_CODE):
        regridded = land_regridder(source_cubes[stash_code])
        native = target_cubes[stash_code]
        assert regridded.shape == native.shape
        for coord_name in ("latitude", "longitude"):
            np.testing.assert_array_equal(regridded.coord(coord_name).points, native.coord(coord_name).points)
            assert regridded.coord(coord_name).coord_system == native.coord(coord_name).coord_system
        for leading_coord in native.coords(dimensions=0, dim_coords=True):
            assert regridded.coord(leading_coord.name()) == source_cubes[stash_code].coord(leading_coord.name())
        assert regridded.attributes["STASH"] == native.attributes["STASH"]
        np.testing.assert_array_equal(
            np.ma.getmaskarray(regridded.data), np.broadcast_to(~target_land_mask, regridded.shape)
        )


@pytest.mark.parametrize("stash_code", UNPROCESSED_STASH_CODES)
def test_unprocessed_land_fields_match_the_native_reconfiguration(
    stash_code, target_name, source_cubes, target_cubes, land_regridder, target_land_mask, points_depending_on_ties
):
    regridded = _flush_denormals(_values(land_regridder(source_cubes[stash_code])))
    native = _values(target_cubes[stash_code])
    points_that_differ = _any_over_leading_dimensions((regridded != native) & target_land_mask, target_land_mask.shape)

    assert not np.any(points_that_differ & ~points_depending_on_ties), (
        f"stash {stash_code} differs from the native {target_name} reconfiguration at "
        f"{np.count_nonzero(points_that_differ & ~points_depending_on_ties)} points that are not exact weight ties"
    )
    if target_name == "n48e":
        assert not np.any(points_that_differ)
    # Ties are rare: well under 0.1% of land points.
    assert np.count_nonzero(points_that_differ) < 1.0e-3 * np.count_nonzero(target_land_mask)


def test_soil_temperature_matches_after_the_lapse_rate_adjustment(
    target_name,
    source_cubes,
    target_cubes,
    source_land_mask_cube,
    target_land_mask_cube,
    target_land_mask,
    land_regridder,
    points_depending_on_ties,
):
    # The interpolated orography the UM compares against is unmasked bilinear.
    unmasked_regridder = build_regridder(
        boolean_mask_cube(source_land_mask_cube, np.ones(source_land_mask_cube.shape)),
        boolean_mask_cube(target_land_mask_cube, np.ones(target_land_mask_cube.shape)),
    )
    interpolated_orography = _values(unmasked_regridder(source_cubes[OROGRAPHY_STASH_CODE]))
    orography_increase = _values(target_cubes[OROGRAPHY_STASH_CODE]) - interpolated_orography

    regridded = (
        _values(land_regridder(source_cubes[SOIL_TEMPERATURE_STASH_CODE]))
        - SOIL_TEMPERATURE_LAPSE_RATE_KELVIN_PER_METRE * orography_increase
    )
    native = _values(target_cubes[SOIL_TEMPERATURE_STASH_CODE])
    differs = (np.abs(regridded - native) > 1.0e-10) & target_land_mask
    points_that_differ = _any_over_leading_dimensions(differs, target_land_mask.shape)

    assert not np.any(points_that_differ & ~points_depending_on_ties)
    if target_name == "n48e":
        assert not np.any(points_that_differ)


def test_soil_moisture_matches_after_the_stress_round_trip(
    source_cubes, target_cubes, source_land_mask_cube, target_land_mask, land_regridder
):
    # The UM interpolates soil moisture stress, (theta - theta_wilt) / (theta_crit - theta_wilt),
    # computed with the source soil parameters, then converts back with the target ones.
    source_soil_moisture_cube = source_cubes[SOIL_MOISTURE_STASH_CODE]
    layer_thicknesses_metres = layer_thicknesses_from_mid_depths(
        source_soil_moisture_cube.coord("depth").points
    )[:, None, None]
    source_wilting_points = _values(source_cubes[WILTING_POINT_STASH_CODE])
    source_critical_points = _values(source_cubes[CRITICAL_POINT_STASH_CODE])
    with np.errstate(divide="ignore", invalid="ignore"):
        source_stress = (
            _values(source_soil_moisture_cube)
            / (WATER_DENSITY_KILOGRAMS_PER_CUBIC_METRE * layer_thicknesses_metres)
            - source_wilting_points
        ) / (source_critical_points - source_wilting_points)
    source_land_mask = mask_from_cube(source_land_mask_cube)
    source_stress = np.where(source_land_mask & np.isfinite(source_stress), source_stress, 0.0)

    target_wilting_points = _values(target_cubes[WILTING_POINT_STASH_CODE])
    target_critical_points = _values(target_cubes[CRITICAL_POINT_STASH_CODE])
    regridded_stress = _values(land_regridder(source_soil_moisture_cube.copy(data=source_stress)))
    regridded = (
        (target_wilting_points + regridded_stress * (target_critical_points - target_wilting_points))
        * WATER_DENSITY_KILOGRAMS_PER_CUBIC_METRE
        * layer_thicknesses_metres
    )
    native = _values(target_cubes[SOIL_MOISTURE_STASH_CODE])
    land_mask = target_land_mask
    relative_differences = np.abs(regridded - native)[:, land_mask] / np.maximum(np.abs(native[:, land_mask]), 1e-12)
    agreement_per_layer = (relative_differences <= 1.0e-9).mean(axis=1)
    assert np.all(agreement_per_layer >= MINIMUM_SOIL_MOISTURE_AGREEMENT), agreement_per_layer
