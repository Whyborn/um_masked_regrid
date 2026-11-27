"""Build and apply regridders between masked domains, on iris cubes or numpy arrays.

A field defined only where the source mask cube is true is projected onto the
points where the target mask cube is true. Source points outside the source
mask can never contribute, and every point inside the target mask always
receives a value. The mask cubes always describe the grids; the data to regrid
can be a cube on the source grid or an array shaped like it.
"""

from dataclasses import dataclass

import iris.coords
import iris.cube
import numpy as np
from scipy.sparse import csr_array

from .coordinates import horizontal_coordinates, horizontal_dimensions, mask_from_cube
from .um_bilinear import um_bilinear_weights
from .weights import nearest_neighbour_distances, valid_point_unit_vectors

# How far a weight row may be from summing to 1 before a builder is rejected.
ROW_SUM_TOLERANCE = 1.0e-9

# How closely a data cube's latitude and longitude must match the source mask
# cube's for it to count as being on the same grid.
COORDINATE_MATCH_TOLERANCE_DEGREES = 1.0e-9


class EmptySourceMaskError(ValueError):
    """The target mask has points to fill but the source mask has none to fill them from."""


@dataclass(frozen=True, eq=False)
class CubeRegridder:
    """Projects fields on the source mask cube's mask onto the target mask cube's mask.

    `weights` has shape `(target_point_count, source_point_count)`, where the
    counts are the numbers of true points in each mask, in C order. Rows are
    non-negative and sum to 1. The mask cubes hold both the masks and the grid
    coordinates. Build with `build_regridder`, then call it on each data cube or
    array: the same regridder serves every field and level that shares this pair
    of masks.
    """

    weights: csr_array
    source_mask_cube: iris.cube.Cube
    target_mask_cube: iris.cube.Cube

    def __call__(self, source_data, *, fill_value=None, source_missing_value=None):
        """Regrid a cube (returning a cube) or an array (returning an array).

        See `regrid_cube` and `regrid_array`.
        """
        regrid_function = regrid_cube if isinstance(source_data, iris.cube.Cube) else regrid_array
        return regrid_function(
            self,
            source_data,
            fill_value=fill_value,
            source_missing_value=source_missing_value,
        )


def _raise_if_source_cannot_fill_target(source_point_count, target_point_count):
    if source_point_count == 0 and target_point_count > 0:
        raise EmptySourceMaskError(
            f"the target mask has {target_point_count} points to fill but the source "
            "mask is empty, so there is nothing to project from"
        )


def _validated_weights(weights, target_point_count, source_point_count):
    weights = csr_array(weights)
    expected_shape = (target_point_count, source_point_count)
    if weights.shape != expected_shape:
        raise ValueError(
            f"weight builder returned shape {weights.shape}, expected {expected_shape}"
        )
    if weights.nnz and weights.data.min() < 0.0:
        raise ValueError("weight builder returned negative weights")
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=ROW_SUM_TOLERANCE):
        raise ValueError("weight builder returned rows that do not sum to 1")
    return weights


def build_regridder(
    source_mask_cube: iris.cube.Cube,
    target_mask_cube: iris.cube.Cube,
    weight_builder=um_bilinear_weights,
) -> CubeRegridder:
    """Build a regridder from the valid source points to the valid target points.

    Each mask cube is 2D (latitude, longitude) with boolean data, true where the
    field is defined; scalar coordinates are allowed. `weight_builder` chooses
    the method. The default, `um_bilinear_weights`, reproduces the UM
    reconfiguration; `nearest_neighbour_weights` and `inverse_distance_weights`
    are alternatives.

    Raises `EmptySourceMaskError` if the target mask has points but the source
    mask has none. An empty target mask is fine and gives a regridder whose
    output is entirely masked (or fill).
    """
    source_mask = mask_from_cube(source_mask_cube)
    target_mask = mask_from_cube(target_mask_cube)
    horizontal_coordinates(source_mask_cube)
    horizontal_coordinates(target_mask_cube)
    source_point_count = int(np.count_nonzero(source_mask))
    target_point_count = int(np.count_nonzero(target_mask))
    _raise_if_source_cannot_fill_target(source_point_count, target_point_count)

    if target_point_count == 0:
        weights = csr_array((0, source_point_count), dtype=np.float64)
    else:
        weights = _validated_weights(
            weight_builder(source_mask_cube, target_mask_cube),
            target_point_count,
            source_point_count,
        )
    return CubeRegridder(
        weights=weights,
        source_mask_cube=source_mask_cube.copy(),
        target_mask_cube=target_mask_cube.copy(),
    )


def _raise_unless_on_source_grid(source_data_cube, source_mask_cube):
    data_coords = horizontal_coordinates(source_data_cube)
    mask_coords = horizontal_coordinates(source_mask_cube)
    for data_coord, mask_coord in zip(data_coords, mask_coords):
        is_same = (
            data_coord.shape == mask_coord.shape
            and data_coord.coord_system == mask_coord.coord_system
            and np.allclose(
                data_coord.points,
                mask_coord.points,
                rtol=0.0,
                atol=COORDINATE_MATCH_TOLERANCE_DEGREES,
            )
        )
        if not is_same:
            raise ValueError(
                f"cube {source_data_cube.name()!r} is not on the source mask cube's grid: "
                f"its {data_coord.name()!r} coordinate differs"
            )


def _source_values_inside_mask(source_data, source_mask, source_missing_value, data_description):
    """The source data at the valid source points, shape `(..., source_point_count)`.

    `data_description` names the data in error messages, e.g. "cube 'x'" or "array".
    """
    is_masked_inside = np.ma.getmaskarray(source_data)[..., source_mask]
    if np.any(is_masked_inside):
        raise ValueError(
            f"{data_description} has {np.count_nonzero(is_masked_inside)} masked "
            "values inside the source mask, so the mask does not describe where the field is defined"
        )
    source_values = np.ma.getdata(source_data)[..., source_mask].astype(np.float64)
    if not np.all(np.isfinite(source_values)):
        raise ValueError(f"{data_description} has non-finite values inside the source mask")
    if source_missing_value is not None and np.any(source_values == source_missing_value):
        raise ValueError(
            f"{data_description} holds the missing value {source_missing_value} "
            "inside the source mask"
        )
    return source_values


def _regridded_values(regridder, source_data, *, fill_value, source_missing_value, data_description):
    """Regrid an array already known to be on the source grid.

    Returns float64 data on the target grid: masked outside the target mask, or
    holding `fill_value` there if it is given.
    """
    source_mask = mask_from_cube(regridder.source_mask_cube)
    target_mask = mask_from_cube(regridder.target_mask_cube)
    source_values = _source_values_inside_mask(
        source_data, source_mask, source_missing_value, data_description
    )
    target_values = _target_values(regridder.weights, source_values)

    leading_shape = source_values.shape[:-1]
    output_values = np.full(
        leading_shape + target_mask.shape,
        0.0 if fill_value is None else fill_value,
        dtype=np.float64,
    )
    output_values[..., target_mask] = target_values
    if fill_value is None:
        output_values = np.ma.masked_array(
            output_values, mask=np.broadcast_to(~target_mask, output_values.shape).copy()
        )
    return output_values


def _target_values(weights, source_values):
    """Apply the weights to every leading-dimension slice at once."""
    leading_shape = source_values.shape[:-1]
    target_point_count, source_point_count = weights.shape
    stacked_field_count = int(np.prod(leading_shape, dtype=np.int64))
    stacked_source_values = source_values.reshape(stacked_field_count, source_point_count)
    stacked_target_values = (weights @ stacked_source_values.T).T
    return stacked_target_values.reshape(leading_shape + (target_point_count,))


def _add_target_horizontal_coordinates(output_cube, target_mask_cube):
    """Give `output_cube` the target mask cube's horizontal coordinates on its last two dimensions."""
    first_horizontal_dimension = output_cube.ndim - 2
    for coord in target_mask_cube.dim_coords:
        (mask_dimension,) = target_mask_cube.coord_dims(coord)
        output_cube.add_dim_coord(coord.copy(), first_horizontal_dimension + mask_dimension)
    for coord in target_mask_cube.aux_coords:
        mask_dimensions = target_mask_cube.coord_dims(coord)
        if mask_dimensions:
            output_cube.add_aux_coord(
                coord.copy(),
                tuple(first_horizontal_dimension + dimension for dimension in mask_dimensions),
            )


def _output_cube(source_data_cube, target_mask_cube, output_data):
    """A cube with the source cube's metadata and leading coordinates on the target grid.

    Anything on the source cube that spans a horizontal dimension describes the
    old grid and is dropped; the target mask cube's horizontal coordinates take
    its place. The mask cube's scalar coordinates belong to the mask, not the
    field, and are not copied.
    """
    horizontal_dimension_set = set(horizontal_dimensions(source_data_cube))

    def is_leading_only(dimensions):
        return not horizontal_dimension_set.intersection(dimensions)

    output_cube = iris.cube.Cube(output_data)
    output_cube.metadata = source_data_cube.metadata
    copied_coords = {}
    for coord in source_data_cube.dim_coords:
        (dimension,) = source_data_cube.coord_dims(coord)
        if is_leading_only((dimension,)):
            copied_coords[id(coord)] = coord.copy()
            output_cube.add_dim_coord(copied_coords[id(coord)], dimension)
    for coord in source_data_cube.aux_coords:
        dimensions = source_data_cube.coord_dims(coord)
        if is_leading_only(dimensions):
            copied_coords[id(coord)] = coord.copy()
            output_cube.add_aux_coord(copied_coords[id(coord)], dimensions)
    for cell_measure in source_data_cube.cell_measures():
        dimensions = source_data_cube.cell_measure_dims(cell_measure)
        if is_leading_only(dimensions):
            output_cube.add_cell_measure(cell_measure.copy(), dimensions)
    for ancillary_variable in source_data_cube.ancillary_variables():
        dimensions = source_data_cube.ancillary_variable_dims(ancillary_variable)
        if is_leading_only(dimensions):
            output_cube.add_ancillary_variable(ancillary_variable.copy(), dimensions)
    for aux_factory in source_data_cube.aux_factories:
        dependencies = [
            dependency for dependency in aux_factory.dependencies.values() if dependency is not None
        ]
        if all(id(dependency) in copied_coords for dependency in dependencies):
            output_cube.add_aux_factory(aux_factory.updated(copied_coords))

    _add_target_horizontal_coordinates(output_cube, target_mask_cube)
    return output_cube


def regrid_cube(
    regridder: CubeRegridder,
    source_data_cube: iris.cube.Cube,
    *,
    fill_value=None,
    source_missing_value=None,
) -> iris.cube.Cube:
    """Project `source_data_cube` onto the target mask. Usually called as `regridder(cube)`.

    The cube's last two dimensions must be latitude and longitude, matching the
    source mask cube's. Leading dimensions (levels, pseudo-levels, times) and
    their coordinates are carried through; the horizontal coordinates are
    replaced by the target mask cube's.

    The output data is float64. By default it is a masked array, masked outside
    the target mask; with `fill_value` it is a plain array holding `fill_value`
    there instead.

    Values outside the source mask are ignored, masked or not. Values inside it
    must be real data: a `ValueError` is raised if any is masked, non-finite or
    equal to `source_missing_value`, since that means the mask does not describe
    where the field is actually defined.
    """
    _raise_unless_on_source_grid(source_data_cube, regridder.source_mask_cube)
    output_values = _regridded_values(
        regridder,
        source_data_cube.data,
        fill_value=fill_value,
        source_missing_value=source_missing_value,
        data_description=f"cube {source_data_cube.name()!r}",
    )
    return _output_cube(source_data_cube, regridder.target_mask_cube, output_values)


def regrid_array(
    regridder: CubeRegridder,
    source_data,
    *,
    fill_value=None,
    source_missing_value=None,
) -> np.ndarray:
    """Project an array onto the target mask. Usually called as `regridder(array)`.

    `source_data` is array-like (numpy, numpy masked or dask) whose last two
    dimensions have the source mask cube's (latitude, longitude) shape. An array
    carries no coordinates, so it is assumed to lie on the source grid; only its
    shape is checked. Leading dimensions are kept, and the last two dimensions
    of the result have the target mask cube's shape.

    `fill_value`, `source_missing_value` and the checks on values inside the
    source mask behave exactly as in `regrid_cube`.
    """
    source_data = np.asanyarray(source_data)
    source_grid_shape = regridder.source_mask_cube.shape
    if source_data.ndim < len(source_grid_shape) or source_data.shape[-2:] != source_grid_shape:
        raise ValueError(
            f"array of shape {source_data.shape} is not on the source grid: its last two "
            f"dimensions must have the source mask cube's shape {source_grid_shape}"
        )
    return _regridded_values(
        regridder,
        source_data,
        fill_value=fill_value,
        source_missing_value=source_missing_value,
        data_description="array",
    )


def nearest_source_distances(regridder: CubeRegridder) -> iris.cube.Cube:
    """Great-circle distance from each valid target point to its nearest valid source point.

    Returns a cube on the target grid, in radians, masked outside the target
    mask. Multiply by the Earth's radius (about 6371 km) for a distance. Useful
    for reporting where values had to be borrowed from far away.
    """
    target_mask = mask_from_cube(regridder.target_mask_cube)
    distances_radians = np.zeros(target_mask.shape, dtype=np.float64)
    if np.any(target_mask):
        nearest_distances_radians, _ = nearest_neighbour_distances(
            valid_point_unit_vectors(regridder.source_mask_cube),
            valid_point_unit_vectors(regridder.target_mask_cube),
            neighbour_count=1,
        )
        distances_radians[target_mask] = nearest_distances_radians[:, 0]
    distance_cube = iris.cube.Cube(
        np.ma.masked_array(distances_radians, mask=~target_mask),
        long_name="distance_to_nearest_valid_source_point",
        units="radians",
    )
    _add_target_horizontal_coordinates(distance_cube, regridder.target_mask_cube)
    return distance_cube
