"""Grid information read from iris cubes, and coordinate transforms on the sphere.

A grid is described by the cube itself: its last two dimensions must have
latitude and longitude dimension coordinates (in that order), optionally on a
rotated pole. The functions here read what the weight builders need from those
coordinates; nothing is copied into a separate grid object.
"""

import iris.coord_systems
import iris.cube
import iris.exceptions
import iris.util
import numpy as np

# Tolerance when deciding whether a longitude axis closes around the globe.
CYCLIC_LONGITUDE_TOLERANCE_DEGREES = 1.0e-9

HORIZONTAL_DIMENSION_COUNT = 2


def horizontal_dimensions(cube: iris.cube.Cube):
    """Indices of the cube's latitude and longitude dimensions (its last two)."""
    if cube.ndim < HORIZONTAL_DIMENSION_COUNT:
        raise ValueError(
            f"cube {cube.name()!r} has {cube.ndim} dimensions; latitude and longitude "
            "must be its last two"
        )
    return cube.ndim - 2, cube.ndim - 1


def mask_from_cube(mask_cube: iris.cube.Cube) -> np.ndarray:
    """The boolean mask held by a 2D (latitude, longitude) mask cube.

    The data must be boolean; masked entries of a masked array count as false.
    """
    if mask_cube.ndim != HORIZONTAL_DIMENSION_COUNT:
        raise ValueError(
            f"mask cube {mask_cube.name()!r} must be 2D (latitude, longitude), "
            f"got shape {mask_cube.shape}; slice it first, e.g. tile_fraction_cube[tile_index]"
        )
    mask_data = mask_cube.data
    if mask_data.dtype != np.bool_:
        raise TypeError(
            f"mask cube {mask_cube.name()!r} must hold boolean data, got dtype {mask_data.dtype}; "
            "convert explicitly, e.g. cube.copy(data=cube.data > 0)"
        )
    return np.ma.getdata(mask_data) & ~np.ma.getmaskarray(mask_data)


def horizontal_coordinates(cube: iris.cube.Cube):
    """The latitude and longitude dimension coordinates of the cube's last two dimensions.

    Returns copies converted to degrees. Raises `ValueError` unless the
    coordinates are latitude-like then longitude-like, strictly increasing, and
    share a coordinate system.
    """
    latitude_dimension, longitude_dimension = horizontal_dimensions(cube)
    try:
        latitude_coord = cube.coord(dimensions=(latitude_dimension,), dim_coords=True)
        longitude_coord = cube.coord(dimensions=(longitude_dimension,), dim_coords=True)
    except iris.exceptions.CoordinateNotFoundError as error:
        raise ValueError(
            f"cube {cube.name()!r} needs dimension coordinates on its last two "
            "dimensions (latitude, longitude)"
        ) from error
    if iris.util.guess_coord_axis(latitude_coord) != "Y" or iris.util.guess_coord_axis(
        longitude_coord
    ) != "X":
        raise ValueError(
            f"the last two dimensions of cube {cube.name()!r} must be latitude then "
            f"longitude, got {latitude_coord.name()!r} and {longitude_coord.name()!r}"
        )
    if latitude_coord.coord_system != longitude_coord.coord_system:
        raise ValueError(
            f"cube {cube.name()!r} has different coordinate systems on latitude and longitude"
        )

    latitude_coord = latitude_coord.copy()
    longitude_coord = longitude_coord.copy()
    latitude_coord.convert_units("degrees")
    longitude_coord.convert_units("degrees")
    for coord in (latitude_coord, longitude_coord):
        if coord.shape[0] > 1 and np.any(np.diff(coord.points) <= 0.0):
            raise ValueError(
                f"coordinate {coord.name()!r} of cube {cube.name()!r} must be strictly increasing"
            )
    if longitude_coord.points[-1] - longitude_coord.points[0] >= 360.0:
        raise ValueError(f"longitude of cube {cube.name()!r} must span less than 360 degrees")
    return latitude_coord, longitude_coord


def rotated_north_pole_degrees(cube: iris.cube.Cube):
    """`(latitude, longitude)` of the true position of a rotated grid's north pole.

    Returns `None` when the grid is not rotated (a `GeogCS` or no coordinate
    system).
    """
    latitude_coord, _ = horizontal_coordinates(cube)
    coord_system = latitude_coord.coord_system
    if not isinstance(coord_system, iris.coord_systems.RotatedGeogCS):
        return None
    if coord_system.north_pole_grid_longitude != 0.0:
        raise NotImplementedError(
            "rotated grids with a non-zero north_pole_grid_longitude are not supported"
        )
    return coord_system.grid_north_pole_latitude, coord_system.grid_north_pole_longitude


def longitude_is_cyclic(longitude_coord) -> bool:
    """True if the longitude coordinate wraps around the globe.

    Either the coordinate is marked `circular`, or stepping once more past the
    last longitude (by the largest spacing along the axis) lands back on the
    first. `longitude_coord` must be in degrees.
    """
    if getattr(longitude_coord, "circular", False):
        return True
    longitudes_degrees = longitude_coord.points
    if longitudes_degrees.size < 2:
        return False
    wrap_gap_degrees = longitudes_degrees[0] + 360.0 - longitudes_degrees[-1]
    largest_spacing_degrees = np.max(np.diff(longitudes_degrees))
    return bool(wrap_gap_degrees <= largest_spacing_degrees + CYCLIC_LONGITUDE_TOLERANCE_DEGREES)


def true_point_coordinates(cube: iris.cube.Cube):
    """True (unrotated) latitude and longitude of every cell centre, in degrees.

    Returns two arrays with the shape of the cube's last two dimensions.
    """
    latitude_coord, longitude_coord = horizontal_coordinates(cube)
    grid_latitudes, grid_longitudes = np.meshgrid(
        latitude_coord.points.astype(np.float64),
        longitude_coord.points.astype(np.float64),
        indexing="ij",
    )
    north_pole = rotated_north_pole_degrees(cube)
    if north_pole is None:
        return grid_latitudes, grid_longitudes
    return unrotate_pole(grid_latitudes, grid_longitudes, *north_pole)


def unrotate_pole(
    rotated_latitudes_degrees,
    rotated_longitudes_degrees,
    north_pole_latitude_degrees,
    north_pole_longitude_degrees,
):
    """Convert rotated-pole coordinates to true latitude and longitude.

    `north_pole_latitude_degrees` and `north_pole_longitude_degrees` are the true
    coordinates of the rotated grid's north pole, as in CF `grid_north_pole_*`
    attributes and iris `RotatedGeogCS`.

    This is the pure coordinate transform: a pole at (90, 180) is the identity
    and a pole at (90, 0) is a 180 degree shift in longitude.

    Returns `(true_latitudes_degrees, true_longitudes_degrees)` with longitudes
    in [0, 360).
    """
    rotated_latitudes = np.radians(np.asarray(rotated_latitudes_degrees, dtype=np.float64))
    rotated_longitudes = np.radians(np.asarray(rotated_longitudes_degrees, dtype=np.float64))
    north_pole_latitude = np.radians(north_pole_latitude_degrees)
    sin_pole_latitude = np.sin(north_pole_latitude)
    cos_pole_latitude = np.cos(north_pole_latitude)

    cos_rotated_latitudes = np.cos(rotated_latitudes)
    sin_rotated_latitudes = np.sin(rotated_latitudes)
    cos_rotated_longitudes = np.cos(rotated_longitudes)
    sin_rotated_longitudes = np.sin(rotated_longitudes)

    sin_true_latitudes = (
        cos_pole_latitude * cos_rotated_latitudes * cos_rotated_longitudes
        + sin_pole_latitude * sin_rotated_latitudes
    )
    true_latitudes_degrees = np.degrees(np.arcsin(np.clip(sin_true_latitudes, -1.0, 1.0)))

    longitude_offsets = np.arctan2(
        cos_rotated_latitudes * sin_rotated_longitudes,
        sin_pole_latitude * cos_rotated_latitudes * cos_rotated_longitudes
        - cos_pole_latitude * sin_rotated_latitudes,
    )
    true_longitudes_degrees = (
        np.degrees(longitude_offsets) + north_pole_longitude_degrees + 180.0
    ) % 360.0
    return true_latitudes_degrees, true_longitudes_degrees


def rotate_pole(
    true_latitudes_degrees,
    true_longitudes_degrees,
    north_pole_latitude_degrees,
    north_pole_longitude_degrees,
):
    """Convert true latitude and longitude to rotated-pole coordinates.

    The inverse of `unrotate_pole`, with the same pole convention. Returns
    `(rotated_latitudes_degrees, rotated_longitudes_degrees)` with longitudes in
    [0, 360).
    """
    # unrotate_pole is a rotation of the sphere; its matrix has the images of the
    # Cartesian basis vectors as columns, and the inverse rotation is its transpose.
    basis_true_latitudes, basis_true_longitudes = unrotate_pole(
        np.array([0.0, 0.0, 90.0]),
        np.array([0.0, 90.0, 0.0]),
        north_pole_latitude_degrees,
        north_pole_longitude_degrees,
    )
    unrotation_matrix = unit_vectors(basis_true_latitudes, basis_true_longitudes).T
    true_latitudes_degrees = np.asarray(true_latitudes_degrees, dtype=np.float64)
    true_vectors = unit_vectors(true_latitudes_degrees, true_longitudes_degrees)
    rotated_vectors = true_vectors @ unrotation_matrix
    rotated_latitudes_degrees = np.degrees(np.arcsin(np.clip(rotated_vectors[:, 2], -1.0, 1.0)))
    rotated_longitudes_degrees = np.degrees(
        np.arctan2(rotated_vectors[:, 1], rotated_vectors[:, 0])
    ) % 360.0
    return (
        rotated_latitudes_degrees.reshape(true_latitudes_degrees.shape),
        rotated_longitudes_degrees.reshape(true_latitudes_degrees.shape),
    )


def unit_vectors(latitudes_degrees, longitudes_degrees):
    """Cartesian unit vectors, shape `(point_count, 3)`, for points on the sphere.

    Inputs of any shape are flattened in C order. Straight-line (chord) distance
    between unit vectors increases monotonically with great-circle distance, so
    nearest-neighbour searches in this space are exact on the sphere and need no
    special handling of the antimeridian or the poles.
    """
    latitudes = np.radians(np.asarray(latitudes_degrees, dtype=np.float64)).ravel()
    longitudes = np.radians(np.asarray(longitudes_degrees, dtype=np.float64)).ravel()
    cos_latitudes = np.cos(latitudes)
    return np.column_stack(
        (
            cos_latitudes * np.cos(longitudes),
            cos_latitudes * np.sin(longitudes),
            np.sin(latitudes),
        )
    )
