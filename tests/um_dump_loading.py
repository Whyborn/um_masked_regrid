"""Load the few fields the dump-based tests need from UM fieldsfiles, as iris cubes."""

import warnings

import iris
import numpy as np

LAND_SEA_MASK_STASH_CODE = 30


def stash_constraint(stash_code):
    return iris.AttributeConstraint(STASH=f"m01s00i{stash_code:03d}")


def load_dump_cubes(dump_path, stash_codes):
    """One cube per stash code, merged over its levels or pseudo-levels by iris.

    Some dumps also hold time-mean copies of a field; the instantaneous one (with
    no cell methods) is used.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded_cubes = iris.load(str(dump_path), [stash_constraint(code) for code in stash_codes])
    cubes_by_stash_code = {}
    for stash_code in stash_codes:
        candidates = [
            cube
            for cube in loaded_cubes.extract(stash_constraint(stash_code))
            if not cube.cell_methods
        ]
        if len(candidates) != 1:
            raise KeyError(
                f"{dump_path} has {len(candidates)} instantaneous cubes with stash code {stash_code}, expected 1"
            )
        cubes_by_stash_code[stash_code] = candidates[0]
    return cubes_by_stash_code


def boolean_mask_cube(template_cube, mask):
    """A 2D mask cube with the horizontal coordinates of `template_cube`."""
    return template_cube.copy(data=np.asarray(mask, dtype=bool))


def land_mask_cube(land_sea_mask_cube):
    return boolean_mask_cube(land_sea_mask_cube, land_sea_mask_cube.data != 0)


def layer_thicknesses_from_mid_depths(mid_depths_metres):
    """Soil layer thicknesses from the depths of the layer mid-points (UM blev)."""
    thicknesses_metres = []
    layer_top_metres = 0.0
    for mid_depth_metres in mid_depths_metres:
        thickness_metres = 2.0 * (mid_depth_metres - layer_top_metres)
        thicknesses_metres.append(thickness_metres)
        layer_top_metres += thickness_metres
    return np.array(thicknesses_metres)
