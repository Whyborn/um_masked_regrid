import functools

import numpy as np
import pytest
from conftest import mask_cube

from um_masked_regrid import (
    inverse_distance_weights,
    nearest_neighbour_distances,
    nearest_neighbour_weights,
    true_point_coordinates,
    unit_vectors,
)

WEIGHT_BUILDERS = [
    pytest.param(nearest_neighbour_weights, id="nearest"),
    pytest.param(inverse_distance_weights, id="inverse_distance"),
    pytest.param(
        functools.partial(inverse_distance_weights, neighbour_count=9, distance_power=1.0),
        id="inverse_distance_k9",
    ),
    pytest.param(
        functools.partial(inverse_distance_weights, maximum_distance_radians=0.05),
        id="inverse_distance_limited",
    ),
]


def build_weights(weight_builder, source_axes, target_axes, source_mask=None, target_mask=None, **options):
    """Call a builder on grids given as (latitudes, longitudes) axes; masks default to all true."""
    return weight_builder(
        mask_cube(*source_axes, mask=source_mask), mask_cube(*target_axes, mask=target_mask), **options
    )


def flattened_true_coordinates(axes):
    latitudes_degrees, longitudes_degrees = true_point_coordinates(mask_cube(*axes))
    return latitudes_degrees.ravel(), longitudes_degrees.ravel()


def haversine_distances_radians(first_latitudes, first_longitudes, second_latitudes, second_longitudes):
    """Great-circle distances between every pair of points, shape (first_count, second_count)."""
    first_latitudes = np.radians(first_latitudes)[:, None]
    first_longitudes = np.radians(first_longitudes)[:, None]
    second_latitudes = np.radians(second_latitudes)[None, :]
    second_longitudes = np.radians(second_longitudes)[None, :]
    half_chord_squared = (
        np.sin((second_latitudes - first_latitudes) / 2.0) ** 2
        + np.cos(first_latitudes) * np.cos(second_latitudes) * np.sin((second_longitudes - first_longitudes) / 2.0) ** 2
    )
    return 2.0 * np.arcsin(np.sqrt(np.clip(half_chord_squared, 0.0, 1.0)))


def random_axes(latitude_count, longitude_count, seed):
    """Irregular axes, deliberately crowded around the antimeridian and both poles."""
    random_generator = np.random.default_rng(seed)
    latitudes = np.concatenate(
        (
            np.degrees(np.arcsin(random_generator.uniform(-1.0, 1.0, latitude_count - 6))),
            random_generator.uniform(86.0, 90.0, 3),
            random_generator.uniform(-90.0, -86.0, 3),
        )
    )
    longitudes = np.concatenate(
        (
            random_generator.uniform(-175.0, 175.0, longitude_count - 6),
            random_generator.uniform(175.0, 180.0, 3),
            random_generator.uniform(-180.0, -175.0, 3),
        )
    )
    return np.sort(latitudes), np.sort(longitudes)


@pytest.fixture
def axis_sets():
    return random_axes(20, 20, seed=10), random_axes(15, 20, seed=11)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_rows_are_non_negative_and_sum_to_one(weight_builder, axis_sets):
    source_axes, target_axes = axis_sets
    weights = build_weights(weight_builder, source_axes, target_axes)
    assert weights.shape == (300, 400)
    assert weights.data.min() > 0.0
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_only_valid_points_get_rows_and_columns():
    weights = build_weights(
        nearest_neighbour_weights,
        ([0.0], [0.0, 1.0, 2.0]),
        ([0.0], [0.9, 1.9]),
        source_mask=[[True, False, True]],
        target_mask=[[False, True]],
    )
    # One valid target (at 1.9) and two valid sources (at 0 and 2): the nearest is the second.
    np.testing.assert_array_equal(weights.toarray(), [[0.0, 1.0]])


def test_nearest_neighbour_matches_brute_force_haversine(axis_sets):
    source_axes, target_axes = axis_sets
    weights = build_weights(nearest_neighbour_weights, source_axes, target_axes)
    source_latitudes, source_longitudes = flattened_true_coordinates(source_axes)
    target_latitudes, target_longitudes = flattened_true_coordinates(target_axes)
    expected_source_indices = np.argmin(
        haversine_distances_radians(target_latitudes, target_longitudes, source_latitudes, source_longitudes), axis=1
    )
    assert weights.nnz == 300
    np.testing.assert_array_equal(weights.indices, expected_source_indices)
    np.testing.assert_array_equal(weights.data, 1.0)


def test_nearest_neighbour_distances_match_haversine(axis_sets):
    source_axes, target_axes = axis_sets
    source_latitudes, source_longitudes = flattened_true_coordinates(source_axes)
    target_latitudes, target_longitudes = flattened_true_coordinates(target_axes)
    distances_radians, source_indices = nearest_neighbour_distances(
        unit_vectors(source_latitudes, source_longitudes),
        unit_vectors(target_latitudes, target_longitudes),
        neighbour_count=5,
    )
    all_distances = haversine_distances_radians(
        target_latitudes, target_longitudes, source_latitudes, source_longitudes
    )
    assert distances_radians.shape == (300, 5)
    assert np.all(np.diff(distances_radians, axis=1) >= 0.0)
    np.testing.assert_allclose(distances_radians, np.sort(all_distances, axis=1)[:, :5], atol=1e-9)
    np.testing.assert_allclose(
        distances_radians, np.take_along_axis(all_distances, source_indices, axis=1), atol=1e-9
    )


def test_nearest_neighbour_crosses_the_antimeridian():
    weights = build_weights(nearest_neighbour_weights, ([0.0], [-170.0, 179.0]), ([0.0], [-179.5]))
    np.testing.assert_array_equal(weights.toarray(), [[0.0, 1.0]])


def test_nearest_neighbour_crosses_the_pole():
    # 89N/0E is 2 degrees from 89N/180E over the pole, but 15 degrees from 74N/0E.
    weights = build_weights(
        nearest_neighbour_weights,
        ([74.0, 89.0], [0.0, 180.0]),
        ([89.0], [0.0]),
        source_mask=[[True, False], [False, True]],
    )
    np.testing.assert_array_equal(weights.toarray(), [[0.0, 1.0]])


def test_inverse_distance_weights_have_expected_values():
    # Sources 1 and 3 degrees either side of the target along the equator.
    source_coordinates = ([0.0], [9.0, 13.0])
    target_coordinates = ([0.0], [10.0])
    squared = build_weights(inverse_distance_weights, source_coordinates, target_coordinates, distance_power=2.0)
    np.testing.assert_allclose(squared.toarray(), [[0.9, 0.1]], atol=1e-12)
    linear = build_weights(inverse_distance_weights, source_coordinates, target_coordinates, distance_power=1.0)
    np.testing.assert_allclose(linear.toarray(), [[0.75, 0.25]], atol=1e-12)


def test_inverse_distance_clips_neighbour_count_to_available_sources():
    weights = build_weights(
        inverse_distance_weights, ([0.0, 1.0], [0.0]), ([0.5, 5.0], [0.0]), neighbour_count=10
    )
    assert weights.shape == (2, 2)
    np.testing.assert_allclose(weights.toarray()[0], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_inverse_distance_coincident_source_takes_all_weight():
    weights = build_weights(
        inverse_distance_weights, ([10.0, 10.5, 11.0], [20.0]), ([10.5], [20.0]), neighbour_count=3
    )
    np.testing.assert_array_equal(weights.toarray(), [[0.0, 1.0, 0.0]])


def test_inverse_distance_maximum_distance_drops_far_neighbours():
    weights = build_weights(
        inverse_distance_weights,
        ([0.0], [1.0, 2.0, 40.0]),
        ([0.0], [0.0]),
        neighbour_count=3,
        maximum_distance_radians=np.radians(5.0),
    )
    np.testing.assert_allclose(weights.toarray(), [[0.8, 0.2, 0.0]], atol=1e-12)


def test_inverse_distance_maximum_distance_falls_back_to_nearest():
    weights = build_weights(
        inverse_distance_weights,
        ([0.0], [30.0, 35.0, 40.0]),
        ([0.0], [0.0]),
        neighbour_count=3,
        maximum_distance_radians=np.radians(5.0),
    )
    np.testing.assert_array_equal(weights.toarray(), [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_no_target_points_gives_zero_rows(weight_builder):
    weights = build_weights(weight_builder, ([0.0, 1.0], [0.0]), ([0.0], [0.0]), target_mask=[[False]])
    assert weights.shape == (0, 2)


@pytest.mark.parametrize("weight_builder", WEIGHT_BUILDERS)
def test_no_source_points_is_an_error(weight_builder):
    with pytest.raises(ValueError, match="zero source points"):
        build_weights(weight_builder, ([0.0], [0.0]), ([0.0], [0.0]), source_mask=[[False]])


def test_malformed_unit_vectors_are_rejected():
    with pytest.raises(ValueError, match=r"\(point_count, 3\)"):
        nearest_neighbour_distances(np.zeros((4, 2)), unit_vectors([0.0], [0.0]), neighbour_count=1)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"neighbour_count": 0}, "neighbour_count"),
        ({"distance_power": 0.0}, "distance_power"),
        ({"maximum_distance_radians": -1.0}, "maximum_distance_radians"),
    ],
)
def test_inverse_distance_rejects_bad_options(options, message):
    coordinates = ([0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match=message):
        build_weights(inverse_distance_weights, coordinates, coordinates, **options)
