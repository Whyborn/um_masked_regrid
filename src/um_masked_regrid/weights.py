"""Weight builders: how valid source points are combined at each target point.

Every builder has the same signature:

    weight_builder(source_mask_cube, target_mask_cube) -> csr_array

where each mask cube is 2D (latitude, longitude) and holds a boolean mask (see
`coordinates.mask_from_cube`). The result has one row per valid target point
and one column per valid source point, each in C order, with non-negative rows
that each sum to 1. Extra options are bound with
`functools.partial`, e.g.

    functools.partial(inverse_distance_weights, neighbour_count=8)

The builders here use only the true positions of the points. The default
builder, which reproduces the UM reconfiguration, is in `um_bilinear`.
"""

import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.spatial import cKDTree

from .coordinates import mask_from_cube, true_point_coordinates, unit_vectors

# Great-circle separation below which a source point is treated as lying exactly
# on the target point (about 6 micrometres on the Earth).
COINCIDENT_DISTANCE_RADIANS = 1.0e-12


def _validated_unit_vectors(unit_vectors_array, description):
    unit_vectors_array = np.asarray(unit_vectors_array, dtype=np.float64)
    if unit_vectors_array.ndim != 2 or unit_vectors_array.shape[1] != 3:
        raise ValueError(
            f"{description} must have shape (point_count, 3), got {unit_vectors_array.shape}"
        )
    return unit_vectors_array


def nearest_neighbour_distances(source_unit_vectors, target_unit_vectors, neighbour_count):
    """Find the nearest source points to each target point.

    Returns `(great_circle_distances_radians, source_indices)`, both of shape
    `(target_point_count, neighbour_count)`, ordered nearest first.
    """
    source_unit_vectors = _validated_unit_vectors(source_unit_vectors, "source_unit_vectors")
    target_unit_vectors = _validated_unit_vectors(target_unit_vectors, "target_unit_vectors")
    source_point_count = source_unit_vectors.shape[0]
    target_point_count = target_unit_vectors.shape[0]
    if source_point_count == 0:
        raise ValueError("cannot search for neighbours among zero source points")
    if not 1 <= neighbour_count <= source_point_count:
        raise ValueError(
            f"neighbour_count must be between 1 and the number of source points "
            f"({source_point_count}), got {neighbour_count}"
        )
    if target_point_count == 0:
        return (
            np.empty((0, neighbour_count), dtype=np.float64),
            np.empty((0, neighbour_count), dtype=np.intp),
        )

    source_tree = cKDTree(source_unit_vectors)
    chord_distances, source_indices = source_tree.query(target_unit_vectors, k=neighbour_count)
    chord_distances = chord_distances.reshape(target_point_count, neighbour_count)
    source_indices = source_indices.reshape(target_point_count, neighbour_count)
    great_circle_distances_radians = 2.0 * np.arcsin(np.clip(chord_distances / 2.0, 0.0, 1.0))
    return great_circle_distances_radians, source_indices


def valid_point_unit_vectors(mask_cube):
    """Unit vectors, shape `(valid point count, 3)`, of the points where the mask cube is true."""
    mask = mask_from_cube(mask_cube)
    latitudes_degrees, longitudes_degrees = true_point_coordinates(mask_cube)
    return unit_vectors(latitudes_degrees[mask], longitudes_degrees[mask])


def nearest_neighbour_weights(source_mask_cube, target_mask_cube):
    """Each valid target point takes the value of its nearest valid source point.

    Values are copied, never blended, so this is the safe choice for flags,
    categories and sets of fields that must stay consistent with one another.
    """
    source_unit_vectors = valid_point_unit_vectors(source_mask_cube)
    _, source_indices = nearest_neighbour_distances(
        source_unit_vectors,
        valid_point_unit_vectors(target_mask_cube),
        neighbour_count=1,
    )
    target_point_count = source_indices.shape[0]
    source_point_count = source_unit_vectors.shape[0]
    return csr_array(
        (
            np.ones(target_point_count, dtype=np.float64),
            (np.arange(target_point_count), source_indices[:, 0]),
        ),
        shape=(target_point_count, source_point_count),
    )


def inverse_distance_weights(
    source_mask_cube,
    target_mask_cube,
    *,
    neighbour_count=4,
    distance_power=2.0,
    maximum_distance_radians=None,
):
    """Each valid target point takes a distance-weighted mean of its nearest valid source points.

    Weights are proportional to `distance ** -distance_power` over the
    `neighbour_count` nearest source points (fewer if there are not that many).

    A source point lying on the target point gets all the weight, so regridding
    onto the same grid and mask reproduces the field exactly.

    If `maximum_distance_radians` is given, neighbours further away than that are
    ignored. The single nearest neighbour is always kept, so isolated target
    points fall back to nearest-neighbour rather than being left unfilled or
    averaging over points that are far apart from each other.
    """
    if neighbour_count < 1:
        raise ValueError(f"neighbour_count must be at least 1, got {neighbour_count}")
    if distance_power <= 0:
        raise ValueError(f"distance_power must be positive, got {distance_power}")
    if maximum_distance_radians is not None and maximum_distance_radians < 0:
        raise ValueError(
            f"maximum_distance_radians must not be negative, got {maximum_distance_radians}"
        )

    source_unit_vectors = valid_point_unit_vectors(source_mask_cube)
    source_point_count = source_unit_vectors.shape[0]
    if source_point_count == 0:
        raise ValueError("cannot search for neighbours among zero source points")
    used_neighbour_count = min(neighbour_count, source_point_count)
    distances_radians, source_indices = nearest_neighbour_distances(
        source_unit_vectors,
        valid_point_unit_vectors(target_mask_cube),
        neighbour_count=used_neighbour_count,
    )
    target_point_count = distances_radians.shape[0]

    is_used = np.ones(distances_radians.shape, dtype=bool)
    if maximum_distance_radians is not None:
        is_used = distances_radians <= maximum_distance_radians
        is_used[:, 0] = True

    with np.errstate(divide="ignore"):
        unnormalised_weights = np.where(is_used, distances_radians**-distance_power, 0.0)

    has_coincident_source = distances_radians[:, 0] <= COINCIDENT_DISTANCE_RADIANS
    unnormalised_weights[has_coincident_source, :] = 0.0
    unnormalised_weights[has_coincident_source, 0] = 1.0

    normalised_weights = unnormalised_weights / unnormalised_weights.sum(axis=1, keepdims=True)

    target_indices = np.repeat(np.arange(target_point_count), used_neighbour_count).reshape(
        target_point_count, used_neighbour_count
    )
    is_nonzero = normalised_weights > 0.0
    return coo_array(
        (
            normalised_weights[is_nonzero],
            (target_indices[is_nonzero], source_indices[is_nonzero]),
        ),
        shape=(target_point_count, source_point_count),
    ).tocsr()
