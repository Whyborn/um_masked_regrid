"""Masked bilinear interpolation that reproduces the UM reconfiguration.

This is the horizontal interpolation the UM reconfiguration applies to land
fields (`h_int_method=1`, `coast_adj_method=3`), generalised from the land-sea
mask to any pair of masks. Each valid target point is handled according to how
many of the four source points around it (its bilinear corners) are valid:

* all four valid ("interior"): ordinary bilinear interpolation;
* one to three valid ("coastal"): the value of the valid corner with the
  largest bilinear weight, copied rather than blended;
* none valid ("unresolved"): the value given to the nearest interior or coastal
  target point, preferring the one nearer the equator, then the western one,
  on exact ties.

The rules were reverse-engineered from native reconfigurations of an N96 dump to
N48 (grids aligned) and N512 (grids not aligned). Field-specific adjustments the
UM makes afterwards, such as lapse-rate corrections for orography changes, are
not part of this module. See the README for the validation results.
"""

import numpy as np
from scipy.sparse import csr_array

from .coordinates import (
    horizontal_coordinates,
    longitude_is_cyclic,
    mask_from_cube,
    rotate_pole,
    rotated_north_pole_degrees,
    true_point_coordinates,
    unit_vectors,
)
from .weights import nearest_neighbour_distances, valid_point_unit_vectors

CORNER_COUNT = 4

# Corner order is (row, column), (row, column + 1), (row + 1, column),
# (row + 1, column + 1): south-west, south-east, north-west, north-east for grids
# whose rows run south to north. When two valid corners have exactly the same
# weight the earlier one in this order is used, which matches the UM in all but
# a handful of tied points (see the README).
CORNER_ROW_OFFSETS = np.array([0, 0, 1, 1])
CORNER_COLUMN_OFFSETS = np.array([0, 1, 0, 1])

# Unresolved points equidistant (to within this) from several resolved points
# take the one nearest the equator, then the western one, matching the UM in
# every tie seen so far. See `_tie_break_ranks`.
DISTANCE_TIE_TOLERANCE_RADIANS = 1.0e-12

# How many nearest resolved points to examine when breaking distance ties.
TIE_SEARCH_NEIGHBOUR_COUNT = 8


def _source_frame_coordinates(source_north_pole, target_latitudes_degrees, target_longitudes_degrees):
    """Target point coordinates in the source grid's own (possibly rotated) frame."""
    if source_north_pole is None:
        return target_latitudes_degrees, target_longitudes_degrees
    return rotate_pole(target_latitudes_degrees, target_longitudes_degrees, *source_north_pole)


def _bracketing_latitudes(latitude_axis_degrees, latitudes_degrees):
    """Lower row index and fractional distance towards the next row.

    Points beyond the first or last row take the edge row's values, which is what
    the UM does for the rows nearest the poles.
    """
    lower_rows = np.clip(
        np.searchsorted(latitude_axis_degrees, latitudes_degrees, side="right") - 1,
        0,
        latitude_axis_degrees.size - 2,
    )
    row_fractions = (latitudes_degrees - latitude_axis_degrees[lower_rows]) / (
        latitude_axis_degrees[lower_rows + 1] - latitude_axis_degrees[lower_rows]
    )
    return lower_rows, np.clip(row_fractions, 0.0, 1.0)


def _bracketing_longitudes(longitude_axis_degrees, longitudes_degrees, longitude_is_cyclic):
    """Left and right column indices and fractional distance from left to right.

    On a cyclic grid the column after the last is the first. On a regional grid,
    points beyond either edge take the edge column's values.
    """
    column_count = longitude_axis_degrees.size
    first_longitude = longitude_axis_degrees[0]
    if longitude_is_cyclic:
        wrapped_longitudes = first_longitude + (longitudes_degrees - first_longitude) % 360.0
        left_columns = np.searchsorted(longitude_axis_degrees, wrapped_longitudes, side="right") - 1
        is_wrapping = left_columns == column_count - 1
        right_columns = np.where(is_wrapping, 0, left_columns + 1)
        right_longitudes = np.where(
            is_wrapping, first_longitude + 360.0, longitude_axis_degrees[right_columns]
        )
        column_fractions = (wrapped_longitudes - longitude_axis_degrees[left_columns]) / (
            right_longitudes - longitude_axis_degrees[left_columns]
        )
        return left_columns, right_columns, column_fractions

    axis_centre = 0.5 * (first_longitude + longitude_axis_degrees[-1])
    wrapped_longitudes = axis_centre + (longitudes_degrees - axis_centre + 180.0) % 360.0 - 180.0
    left_columns = np.clip(
        np.searchsorted(longitude_axis_degrees, wrapped_longitudes, side="right") - 1,
        0,
        column_count - 2,
    )
    right_columns = left_columns + 1
    column_fractions = (wrapped_longitudes - longitude_axis_degrees[left_columns]) / (
        longitude_axis_degrees[right_columns] - longitude_axis_degrees[left_columns]
    )
    return left_columns, right_columns, np.clip(column_fractions, 0.0, 1.0)


def bilinear_corners(source_cube, target_latitudes_degrees, target_longitudes_degrees):
    """Bilinear stencils on the source cube's grid for a set of target points.

    `source_cube` is any cube on the source grid; only its latitude and
    longitude coordinates are used, and each needs at least two points. Target
    coordinates are true latitude and longitude, of any shape; they are
    flattened in C order. Returns `(corner_rows, corner_columns, corner_weights)`,
    each of shape `(4, target point count)`, with corners in the order described
    by `CORNER_ROW_OFFSETS` and weights that are linear in degrees and sum to 1.
    """
    latitude_coord, longitude_coord = horizontal_coordinates(source_cube)
    if latitude_coord.shape[0] < 2 or longitude_coord.shape[0] < 2:
        raise ValueError(
            "bilinear interpolation needs at least two source points along latitude and longitude"
        )
    frame_latitudes, frame_longitudes = _source_frame_coordinates(
        rotated_north_pole_degrees(source_cube),
        np.asarray(target_latitudes_degrees, dtype=np.float64).ravel(),
        np.asarray(target_longitudes_degrees, dtype=np.float64).ravel(),
    )
    lower_rows, row_fractions = _bracketing_latitudes(
        latitude_coord.points.astype(np.float64), frame_latitudes
    )
    left_columns, right_columns, column_fractions = _bracketing_longitudes(
        longitude_coord.points.astype(np.float64),
        frame_longitudes,
        longitude_is_cyclic(longitude_coord),
    )
    corner_rows = lower_rows[None, :] + CORNER_ROW_OFFSETS[:, None]
    corner_columns = np.where(
        CORNER_COLUMN_OFFSETS[:, None] == 0, left_columns[None, :], right_columns[None, :]
    )
    corner_weights = np.stack(
        (
            (1.0 - row_fractions) * (1.0 - column_fractions),
            (1.0 - row_fractions) * column_fractions,
            row_fractions * (1.0 - column_fractions),
            row_fractions * column_fractions,
        )
    )
    return corner_rows, corner_columns, corner_weights


def _valid_point_numbers(mask):
    """Map each grid point (flattened, C order) to its position among the valid points, or -1."""
    valid_point_numbers = np.full(mask.size, -1, dtype=np.int64)
    valid_point_numbers[np.flatnonzero(mask)] = np.arange(np.count_nonzero(mask))
    return valid_point_numbers


def _rows_to_csr(row_entry_counts, entry_columns, entry_weights, column_count):
    """Assemble a CSR array from row-ordered entries, keeping their order within each row.

    Entry order is kept so that interpolated values are summed in corner order,
    which keeps results bitwise identical to the UM's.
    """
    index_pointers = np.concatenate(([0], np.cumsum(row_entry_counts)))
    return csr_array(
        (entry_weights, entry_columns, index_pointers),
        shape=(row_entry_counts.size, column_count),
    )


def _tie_break_ranks(target_mask_cube, target_mask):
    """Preference order of the valid target points when breaking distance ties.

    Points nearer the equator come first, then points earlier in C order (the
    western one of two points in the same row). Latitude is measured in the
    target grid's own (possibly rotated) frame.
    """
    target_latitude_coord, _ = horizontal_coordinates(target_mask_cube)
    row_numbers = np.nonzero(target_mask)[0]
    frame_latitudes_degrees = target_latitude_coord.points[row_numbers]
    preference_order = np.lexsort(
        (np.arange(frame_latitudes_degrees.size), np.abs(frame_latitudes_degrees))
    )
    ranks = np.empty(preference_order.size, dtype=np.int64)
    ranks[preference_order] = np.arange(preference_order.size)
    return ranks


def _nearest_preferred(candidate_unit_vectors, candidate_ranks, query_unit_vectors):
    """Index of the nearest candidate to each query; exact ties go to the lowest rank."""
    neighbour_count = min(TIE_SEARCH_NEIGHBOUR_COUNT, candidate_unit_vectors.shape[0])
    distances_radians, candidate_indices = nearest_neighbour_distances(
        candidate_unit_vectors, query_unit_vectors, neighbour_count
    )
    is_tied_with_nearest = (
        distances_radians - distances_radians[:, :1] <= DISTANCE_TIE_TOLERANCE_RADIANS
    )
    tied_ranks = np.where(
        is_tied_with_nearest, candidate_ranks[candidate_indices], np.iinfo(np.int64).max
    )
    return np.take_along_axis(
        candidate_indices, np.argmin(tied_ranks, axis=1)[:, None], axis=1
    )[:, 0]


def um_bilinear_weights(source_mask_cube, target_mask_cube):
    """Weights reproducing the UM reconfiguration's masked bilinear interpolation.

    See the module docstring for how interior, coastal and unresolved target
    points are treated. Interpolation happens in the source grid's own frame, so
    the target may be on a different (or differently rotated) grid.

    If no target point has any valid bilinear corner, there are no resolved
    points to copy from, and each target point instead takes its nearest valid
    source point. The UM never meets this case for land fields.
    """
    source_mask = mask_from_cube(source_mask_cube)
    target_mask = mask_from_cube(target_mask_cube)
    target_grid_latitudes_degrees, target_grid_longitudes_degrees = true_point_coordinates(
        target_mask_cube
    )
    target_latitudes_degrees = target_grid_latitudes_degrees[target_mask]
    target_longitudes_degrees = target_grid_longitudes_degrees[target_mask]
    corner_rows, corner_columns, corner_weights = bilinear_corners(
        source_mask_cube, target_latitudes_degrees, target_longitudes_degrees
    )
    source_valid_point_numbers = _valid_point_numbers(source_mask)
    source_point_count = int(np.count_nonzero(source_mask))
    corner_source_numbers = source_valid_point_numbers[
        np.ravel_multi_index((corner_rows, corner_columns), source_mask.shape)
    ]
    corner_is_valid = corner_source_numbers >= 0
    valid_corner_counts = corner_is_valid.sum(axis=0)
    is_interior = valid_corner_counts == CORNER_COUNT
    is_resolved = valid_corner_counts > 0

    # Coastal points copy their heaviest valid corner; np.argmax takes the first
    # corner among exact ties.
    heaviest_valid_corners = np.argmax(np.where(corner_is_valid, corner_weights, -1.0), axis=0)
    resolved_point_numbers = np.flatnonzero(is_resolved)
    resolved_is_interior = is_interior[resolved_point_numbers]

    # Interior rows keep their non-zero corner weights in corner order; coastal
    # rows have a single entry of weight 1.
    interior_entry_is_kept = (corner_weights > 0.0) & is_interior[None, :]
    coastal_entry_is_kept = (
        np.arange(CORNER_COUNT)[:, None] == heaviest_valid_corners[None, :]
    ) & (is_resolved & ~is_interior)[None, :]
    entry_is_kept = (interior_entry_is_kept | coastal_entry_is_kept)[:, resolved_point_numbers]
    entry_weights = np.where(
        resolved_is_interior[None, :], corner_weights[:, resolved_point_numbers], 1.0
    )
    # Transpose so entries are grouped by target point, then by corner.
    resolved_weights = _rows_to_csr(
        row_entry_counts=entry_is_kept.sum(axis=0),
        entry_columns=corner_source_numbers[:, resolved_point_numbers].T[entry_is_kept.T],
        entry_weights=entry_weights.T[entry_is_kept.T],
        column_count=source_point_count,
    )

    target_point_count = target_latitudes_degrees.size
    unresolved_point_numbers = np.flatnonzero(~is_resolved)
    if unresolved_point_numbers.size == 0:
        return resolved_weights

    target_unit_vectors = unit_vectors(target_latitudes_degrees, target_longitudes_degrees)
    if resolved_point_numbers.size == 0:
        _, nearest_source_numbers = nearest_neighbour_distances(
            valid_point_unit_vectors(source_mask_cube),
            target_unit_vectors,
            neighbour_count=1,
        )
        return _rows_to_csr(
            row_entry_counts=np.ones(target_point_count, dtype=np.int64),
            entry_columns=nearest_source_numbers[:, 0],
            entry_weights=np.ones(target_point_count),
            column_count=source_point_count,
        )

    # Each target point's row in the final matrix is a row of resolved_weights:
    # its own if resolved, its nearest resolved neighbour's if not.
    resolved_row_of_target = np.empty(target_point_count, dtype=np.int64)
    resolved_row_of_target[resolved_point_numbers] = np.arange(resolved_point_numbers.size)
    resolved_row_of_target[unresolved_point_numbers] = _nearest_preferred(
        target_unit_vectors[resolved_point_numbers],
        _tie_break_ranks(target_mask_cube, target_mask)[resolved_point_numbers],
        target_unit_vectors[unresolved_point_numbers],
    )
    return resolved_weights[resolved_row_of_target]
