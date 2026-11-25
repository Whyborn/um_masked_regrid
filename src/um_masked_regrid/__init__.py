"""
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

Regrid fields that are only defined on part of a grid (e.g. CABLE per-tile fields), on iris cubes.
"""

from . import _version
from .coordinates import (
    horizontal_coordinates,
    longitude_is_cyclic,
    mask_from_cube,
    rotate_pole,
    rotated_north_pole_degrees,
    true_point_coordinates,
    unit_vectors,
    unrotate_pole,
)
from .regridder import (
    CubeRegridder,
    EmptySourceMaskError,
    build_regridder,
    nearest_source_distances,
    regrid_cube,
)
from .um_bilinear import bilinear_corners, um_bilinear_weights
from .weights import (
    inverse_distance_weights,
    nearest_neighbour_distances,
    nearest_neighbour_weights,
    valid_point_unit_vectors,
)

__all__ = [
    "CubeRegridder",
    "EmptySourceMaskError",
    "bilinear_corners",
    "build_regridder",
    "horizontal_coordinates",
    "inverse_distance_weights",
    "longitude_is_cyclic",
    "mask_from_cube",
    "nearest_neighbour_distances",
    "nearest_neighbour_weights",
    "nearest_source_distances",
    "regrid_cube",
    "rotate_pole",
    "rotated_north_pole_degrees",
    "true_point_coordinates",
    "um_bilinear_weights",
    "unit_vectors",
    "unrotate_pole",
    "valid_point_unit_vectors",
]

__version__ = _version.get_versions()["version"]
