"""Shared cube builders."""

import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import pytest

from um_masked_regrid import true_point_coordinates

EARTH_RADIUS_METRES = 6371229.0


def mask_cube(latitudes_degrees, longitudes_degrees, mask=None, north_pole=None, circular=False):
    """A 2D mask cube on the given axes, all true unless `mask` is given.

    `north_pole` is the true `(latitude, longitude)` of a rotated grid's pole;
    the axes are then rotated coordinates.
    """
    latitudes_degrees = np.asarray(latitudes_degrees, dtype=np.float64)
    longitudes_degrees = np.asarray(longitudes_degrees, dtype=np.float64)
    if north_pole is None:
        coord_system = iris.coord_systems.GeogCS(EARTH_RADIUS_METRES)
        latitude_name, longitude_name = "latitude", "longitude"
    else:
        coord_system = iris.coord_systems.RotatedGeogCS(
            *north_pole, ellipsoid=iris.coord_systems.GeogCS(EARTH_RADIUS_METRES)
        )
        latitude_name, longitude_name = "grid_latitude", "grid_longitude"
    latitude_coord = iris.coords.DimCoord(
        latitudes_degrees, standard_name=latitude_name, units="degrees", coord_system=coord_system
    )
    longitude_coord = iris.coords.DimCoord(
        longitudes_degrees,
        standard_name=longitude_name,
        units="degrees",
        coord_system=coord_system,
        circular=circular,
    )
    shape = (latitudes_degrees.size, longitudes_degrees.size)
    mask = np.ones(shape, dtype=bool) if mask is None else np.asanyarray(mask)
    return iris.cube.Cube(
        mask,
        long_name="mask",
        dim_coords_and_dims=[(latitude_coord, 0), (longitude_coord, 1)],
    )


def regular_mask_cube(
    first_latitude_degrees,
    first_longitude_degrees,
    latitude_spacing_degrees,
    longitude_spacing_degrees,
    row_count,
    column_count,
    north_pole=None,
    circular=False,
):
    """An all-true mask cube on an evenly spaced grid, given its first cell centre and spacings."""
    return mask_cube(
        first_latitude_degrees + latitude_spacing_degrees * np.arange(row_count),
        first_longitude_degrees + longitude_spacing_degrees * np.arange(column_count),
        north_pole=north_pole,
        circular=circular,
    )


def global_mask_cube(spacing_degrees):
    """An all-true global regular grid with cell centres offset half a cell from the poles."""
    row_count = round(180.0 / spacing_degrees)
    column_count = round(360.0 / spacing_degrees)
    return regular_mask_cube(
        -90.0 + spacing_degrees / 2.0,
        spacing_degrees / 2.0,
        spacing_degrees,
        spacing_degrees,
        row_count,
        column_count,
        circular=True,
    )


def with_mask(grid_cube, mask):
    """A copy of a mask cube with different mask data."""
    return grid_cube.copy(data=np.asanyarray(mask))


def field_cube(grid_cube, values, leading_coords=()):
    """A data cube holding `values` on the grid of `grid_cube`.

    `values` has shape `(..., *grid_cube.shape)`. `leading_coords` are
    `(DimCoord, dimension)` pairs for the leading dimensions.
    """
    if not hasattr(values, "ndim"):  # leave numpy, masked and dask arrays as they are
        values = np.asanyarray(values)
    leading_dimension_count = values.ndim - 2
    dim_coords_and_dims = list(leading_coords) + [
        (coord.copy(), leading_dimension_count + grid_cube.coord_dims(coord)[0])
        for coord in grid_cube.dim_coords
    ]
    return iris.cube.Cube(
        values,
        long_name="test_field",
        units="K",
        dim_coords_and_dims=dim_coords_and_dims,
    )


def regrid_values(regridder, values, fill_value=np.nan, source_missing_value=None):
    """Regrid a plain array on the regridder's source grid, returning a plain array."""
    regridded_cube = regridder(
        field_cube(regridder.source_mask_cube, values),
        fill_value=fill_value,
        source_missing_value=source_missing_value,
    )
    return regridded_cube.data


def true_latitudes(cube):
    """True latitudes of every point of a cube, as a 2D array."""
    return true_point_coordinates(cube)[0]


def true_longitudes(cube):
    """True longitudes of every point of a cube, as a 2D array."""
    return true_point_coordinates(cube)[1]


def smooth_field(cube):
    """An analytic field that is smooth everywhere on the sphere."""
    latitudes_degrees, longitudes_degrees = true_point_coordinates(cube)
    latitudes = np.radians(latitudes_degrees)
    longitudes = np.radians(longitudes_degrees)
    return 280.0 + 20.0 * np.cos(latitudes) * np.sin(2.0 * longitudes) + 10.0 * np.sin(latitudes)


def random_mask(cube, valid_fraction, seed):
    """A reproducible random mask, true at about `valid_fraction` of the cube's points."""
    return np.random.default_rng(seed).random(cube.shape) < valid_fraction


@pytest.fixture
def coarse_grid():
    """An all-true global 5 degree grid (36x72)."""
    return global_mask_cube(5.0)


@pytest.fixture
def fine_grid():
    """An all-true global 2 degree grid (90x180)."""
    return global_mask_cube(2.0)
