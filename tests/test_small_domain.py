"""End-to-end regrids of a small regional domain, with masked data.

The source is a 20x20 grid covering 0-20 N and 100-120 E. It is regridded onto
a 40x40 grid of the same domain, and downscaled onto a 20x20 grid of a smaller
domain nested inside it. Grids put cell centres half a cell in from their domain
edges, so the 40x40 grid's outermost points lie beyond the source's and are
clamped to its edge rows and columns.

The masks are circular "continents". The target's continent is larger than the
source's and it has an island the source lacks, so the target has interior
points (four valid source corners), coastal points (one to three) and
unresolved points (none). The source data is masked outside the source mask,
with a huge value beneath the mask that would show up if it leaked into the
result.
"""

import iris.coords
import numpy as np
from conftest import field_cube, mask_cube, true_latitudes, true_longitudes

from um_masked_regrid import bilinear_corners, build_regridder, mask_from_cube

# (south, north, west, east) in degrees.
SOURCE_DOMAIN_DEGREES = (0.0, 20.0, 100.0, 120.0)
NESTED_DOMAIN_DEGREES = (5.0, 15.0, 105.0, 115.0)
VALUE_BENEATH_MASK = -1.0e20
LEVEL_OFFSET = 10.0


def domain_grid(domain_degrees, points_per_side, land_circles):
    """A mask cube of `points_per_side` squared cells covering `domain_degrees`.

    The mask is true inside any of `land_circles`, given as
    `((centre_latitude, centre_longitude), radius)` in degrees.
    """
    south, north, west, east = domain_degrees
    cell_offsets = np.arange(points_per_side) + 0.5
    latitudes = south + (north - south) / points_per_side * cell_offsets
    longitudes = west + (east - west) / points_per_side * cell_offsets
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    mask = np.zeros(latitude_grid.shape, dtype=bool)
    for (centre_latitude, centre_longitude), radius in land_circles:
        mask |= (latitude_grid - centre_latitude) ** 2 + (longitude_grid - centre_longitude) ** 2 < radius**2
    return mask_cube(latitudes, longitudes, mask)


def linear_field(latitudes, longitudes):
    """A field that bilinear interpolation reproduces exactly."""
    return 300.0 + 2.0 * latitudes - 0.5 * longitudes


def masked_two_level_source_cube(source_mask_cube):
    """Two levels of the linear field, masked outside the source mask."""
    source_mask = mask_from_cube(source_mask_cube)
    single_level = linear_field(true_latitudes(source_mask_cube), true_longitudes(source_mask_cube))
    values = np.stack([single_level, single_level + LEVEL_OFFSET])
    values[:, ~source_mask] = VALUE_BENEATH_MASK
    masked_values = np.ma.masked_array(values, mask=np.broadcast_to(~source_mask, values.shape))
    level_coord = iris.coords.DimCoord([1, 2], standard_name="model_level_number", units="1")
    return field_cube(source_mask_cube, masked_values, leading_coords=[(level_coord, 0)])


def great_circle_distances(latitudes_degrees, longitudes_degrees, latitude_degrees, longitude_degrees):
    latitudes, longitudes = np.radians(latitudes_degrees), np.radians(longitudes_degrees)
    latitude, longitude = np.radians(latitude_degrees), np.radians(longitude_degrees)
    cosine = np.sin(latitudes) * np.sin(latitude) + np.cos(latitudes) * np.cos(latitude) * np.cos(
        longitudes - longitude
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def check_regrid_follows_the_um_rules(source_mask_cube, target_mask_cube):
    source_mask = mask_from_cube(source_mask_cube)
    target_mask = mask_from_cube(target_mask_cube)
    source_cube = masked_two_level_source_cube(source_mask_cube)

    regridded_cube = build_regridder(source_mask_cube, target_mask_cube)(source_cube)

    # The output is on the target grid, keeps the level coordinate, and is masked exactly outside the target mask.
    assert regridded_cube.shape == (2, *target_mask.shape)
    for coord_name in ("latitude", "longitude"):
        np.testing.assert_array_equal(
            regridded_cube.coord(coord_name).points, target_mask_cube.coord(coord_name).points
        )
    np.testing.assert_array_equal(regridded_cube.coord("model_level_number").points, [1, 2])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(regridded_cube.data), np.broadcast_to(~target_mask, regridded_cube.shape)
    )

    # Every level is regridded with the same weights.
    lower_level, upper_level = (np.ma.getdata(regridded_cube.data[level])[target_mask] for level in (0, 1))
    np.testing.assert_allclose(upper_level - lower_level, LEVEL_OFFSET, rtol=0.0, atol=1e-9)

    # Nothing from beneath the source mask leaks in.
    valid_source_values = np.ma.getdata(source_cube.data[0])[source_mask]
    assert lower_level.min() >= valid_source_values.min()
    assert lower_level.max() <= valid_source_values.max()

    # Classify valid target points by how many of their bilinear corners are valid.
    target_latitudes = true_latitudes(target_mask_cube)[target_mask]
    target_longitudes = true_longitudes(target_mask_cube)[target_mask]
    rows, columns, weights = bilinear_corners(source_mask_cube, target_latitudes, target_longitudes)
    valid_corners = source_mask[rows, columns]
    valid_corner_count = valid_corners.sum(axis=0)
    is_interior = valid_corner_count == 4
    is_coastal = (valid_corner_count > 0) & (valid_corner_count < 4)
    is_unresolved = valid_corner_count == 0
    assert is_interior.any() and is_coastal.any() and is_unresolved.any()

    # Interior: bilinear interpolation, so the linear field is exact. Points beyond the
    # outermost source rows and columns take the values at the edge.
    source_latitude_points = source_mask_cube.coord("latitude").points
    source_longitude_points = source_mask_cube.coord("longitude").points
    clamped_latitudes = np.clip(target_latitudes, source_latitude_points[0], source_latitude_points[-1])
    clamped_longitudes = np.clip(target_longitudes, source_longitude_points[0], source_longitude_points[-1])
    np.testing.assert_allclose(
        lower_level[is_interior],
        linear_field(clamped_latitudes, clamped_longitudes)[is_interior],
        rtol=0.0,
        atol=1e-9,
    )

    # Coastal: a copy of the valid corner with the largest weight, earliest corner on ties.
    source_values = np.ma.getdata(source_cube.data[0])
    heaviest_valid_corner = np.argmax(np.where(valid_corners, weights, -1.0), axis=0)
    point_indices = np.arange(target_latitudes.size)
    heaviest_corner_values = source_values[
        rows[heaviest_valid_corner, point_indices], columns[heaviest_valid_corner, point_indices]
    ]
    np.testing.assert_array_equal(lower_level[is_coastal], heaviest_corner_values[is_coastal])

    # Unresolved: a copy of one of the nearest interior or coastal target points.
    is_resolved = ~is_unresolved
    for unresolved_index in np.flatnonzero(is_unresolved):
        distances = great_circle_distances(
            target_latitudes[is_resolved],
            target_longitudes[is_resolved],
            target_latitudes[unresolved_index],
            target_longitudes[unresolved_index],
        )
        nearest_values = lower_level[is_resolved][np.isclose(distances, distances.min(), rtol=1e-12, atol=0.0)]
        assert lower_level[unresolved_index] in nearest_values


def test_masked_20_by_20_domain_onto_a_40_by_40_grid_of_the_same_domain():
    source_mask_cube = domain_grid(SOURCE_DOMAIN_DEGREES, 20, [((10.0, 110.0), 6.0)])
    target_mask_cube = domain_grid(SOURCE_DOMAIN_DEGREES, 40, [((10.0, 110.0), 7.0), ((3.0, 116.0), 1.3)])
    check_regrid_follows_the_um_rules(source_mask_cube, target_mask_cube)


def test_masked_20_by_20_domain_downscaled_onto_a_nested_20_by_20_domain():
    source_mask_cube = domain_grid(SOURCE_DOMAIN_DEGREES, 20, [((12.0, 112.0), 5.0)])
    target_mask_cube = domain_grid(NESTED_DOMAIN_DEGREES, 20, [((12.0, 112.0), 6.0), ((6.5, 106.5), 1.0)])
    check_regrid_follows_the_um_rules(source_mask_cube, target_mask_cube)
