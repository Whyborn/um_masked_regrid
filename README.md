# um_masked_regrid

Regrids fields that are only defined on part of a grid, such as CABLE per-tile
fields, which the UM reconfiguration cannot handle.

- **What it does**: given a source and a target mask cube, `um_masked_regrid` builds a
  regridder. Calling it on a cube projects a field defined on the source mask onto
  the target mask and returns a cube on the target grid.
- **Matches the UM**: the default method reproduces the UM reconfiguration's
  horizontal interpolation of land fields. With the land-sea mask as the mask, it
  is bitwise identical to native N96 → N48 output. For N96 → N512 it differs only
  at 16 of 280,593 land points, where two corners tie exactly in weight.
- **Not included**: field-specific adjustments the UM makes after interpolating,
  such as the soil temperature lapse-rate correction and the soil-moisture stress
  conversion, and filling points outside the target mask. These belong to the
  script that reconfigures the dump.
- **Guarantees**: source points outside the source mask never contribute, and
  every point inside the target mask gets a value.
- **Cubes**: the last two dimensions must be latitude and longitude, with
  dimension coordinates (rotated poles are supported). Leading dimensions such as
  soil levels or tiles are carried through.
- **Dependencies**: Python 3.10 or later, `numpy`, `scipy` and `iris`.
- **Why not ANTS**: regridding with ANTS and then filling missing points with
  `ants.analysis.make_consistent_with_lsm` differs from the UM at most coastal
  points; see [Comparison with ANTS](#comparison-with-ants).

```python
import iris
from um_masked_regrid import build_regridder

source_tile_fractions = iris.load_cube("source_dump", iris.AttributeConstraint(STASH="m01s00i216"))
target_tile_fractions = iris.load_cube("target_dump", iris.AttributeConstraint(STASH="m01s00i216"))

# One regridder per tile: the mask cubes say where that tile exists on each grid.
tile_index = 0
source_tile = source_tile_fractions[tile_index]
target_tile = target_tile_fractions[tile_index]
regridder = build_regridder(
    source_tile.copy(data=source_tile.data > 0),
    target_tile.copy(data=target_tile.data > 0),
)

# Reuse it for every field and level on that tile.
target_soil_temperature = regridder(source_soil_temperature_on_tile)
```

Install from a clone with `pip install .`, or `pip install -e ".[dev]"` for
development. Run the tests with `python -m pytest`.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter)
and the [`ACCESS-NRI/cookiecutter-pypackage-access`](https://github.com/ACCESS-NRI/cookiecutter-pypackage-access)
project template.

## Details

### Layout

| Path | Contents |
| --- | --- |
| `src/um_masked_regrid/regridder.py` | `build_regridder`, `CubeRegridder` and `nearest_source_distances` |
| `src/um_masked_regrid/um_bilinear.py` | `um_bilinear_weights`, the default method, which reproduces the UM reconfiguration |
| `src/um_masked_regrid/weights.py` | `nearest_neighbour_weights` and `inverse_distance_weights` |
| `src/um_masked_regrid/coordinates.py` | Functions that read masks and grid coordinates from cubes, and pole rotation |
| `tests/` | Unit tests on synthetic cubes, including small regional regrids to a finer and a coarser grid |

### How it works

`build_regridder(source_mask_cube, target_mask_cube)` computes a sparse matrix of
weights with one row per valid target point and one column per valid source
point. Rows are non-negative and sum to 1, so each output is a weighted mean of
valid source values and always lies within their range. Points outside the
source mask have no column, which is what guarantees they cannot leak into the
result.

The returned `CubeRegridder` holds the weights and copies of both mask cubes,
which carry the masks and the grid coordinates. Calling it on a data cube is a
single sparse matrix product over all leading dimensions at once.

Nearest-point searches use a k-d tree over unit vectors on the sphere. Straight
line distance between unit vectors increases with great-circle distance, so the
search is exact on the sphere, and the antimeridian and poles need no special
treatment.

### Cubes

**Mask cubes** are 2D (latitude, longitude) with boolean data, true where the
field is defined. Scalar coordinates are allowed. Build them explicitly, e.g.
`land_cube.copy(data=land_cube.data != 0)`; anything other than boolean data is
rejected. Masked entries of a masked mask count as false. Slice per-tile cubes
first, e.g. `tile_fraction_cube[tile_index]`.

**Grid coordinates** are read from the dimension coordinates of the last two
dimensions:

- They must be latitude then longitude, strictly increasing, in any angular
  units.
- A `RotatedGeogCS` gives a rotated grid; the coordinates are then rotated
  latitude and longitude. A non-zero `north_pole_grid_longitude` is not
  supported.
- Longitude wraps around the globe if the coordinate is `circular`, or if one
  more grid spacing closes 360°.
- Longitudes may be in 0 to 360 or -180 to 180.

**Data cubes** passed to the regridder must be on the source mask cube's grid
(same latitude, longitude and coordinate system); anything else raises
`ValueError`. Any number of leading dimensions is allowed.

**The output cube** is built as follows.

| Taken from | What is carried over |
| --- | --- |
| Data cube | Names, units, attributes and cell methods |
| Data cube | Coordinates, cell measures and ancillary variables on leading dimensions only, plus scalar coordinates |
| Data cube | Aux factories, if none of their dependencies span a horizontal dimension |
| Target mask cube | Latitude and longitude dimension coordinates |
| Target mask cube | Auxiliary coordinates spanning only its horizontal dimensions (e.g. 2D true latitude on a rotated grid) |
| Neither | Anything on the data cube spanning a horizontal dimension, and the target mask cube's scalar coordinates |

The data is float64:

- by default, a masked array masked outside the target mask (as iris loads
  land-packed fields);
- with `regridder(cube, fill_value=...)`, a plain array holding `fill_value`
  outside the target mask.

### Choosing a method

| Builder | Behaviour | Use for |
| --- | --- | --- |
| `um_bilinear_weights` (default) | Masked bilinear interpolation, as the UM reconfiguration does it | Anything that should match what the UM would do |
| `nearest_neighbour_weights` | Copies the nearest valid source value | Flags, categories, and groups of fields that must stay mutually consistent (e.g. snow layer depth, mass and count) |
| `inverse_distance_weights` | Mean of the nearest valid source values, weighted by `distance ** -distance_power` | Smooth fields where a smoother result than nearest neighbour is wanted |

Pass a builder as the third argument, `build_regridder(source_mask_cube,
target_mask_cube, nearest_neighbour_weights)`. Options for
`inverse_distance_weights` are bound with `functools.partial`:

- `neighbour_count` (default 4): reduced automatically if fewer valid source
  points exist.
- `distance_power` (default 2).
- `maximum_distance_radians` (default none): neighbours further away are ignored,
  but the nearest one is always kept.

To add a method, write a function
`(source_mask_cube, target_mask_cube) -> scipy.sparse.csr_array` with one row per
valid target point and one column per valid source point, each in C order.
`mask_from_cube` and `true_point_coordinates` give the masks and point
positions. `build_regridder` rejects builders whose rows are negative or do not
sum to 1.

### The UM-compatible method

For each valid target point, `um_bilinear_weights` finds the four source points
around it (its bilinear corners). It then acts on how many of them are valid:

| Valid corners | Name | Value given to the target point |
| --- | --- | --- |
| 4 | interior | Bilinear interpolation, weights linear in degrees |
| 1–3 | coastal | The value of the valid corner with the largest bilinear weight, copied rather than blended |
| 0 | unresolved | The value given to the nearest interior or coastal target point (great-circle distance) |

Further rules:

- **Edge rows and columns**: target points beyond the first or last source row
  (e.g. N512's polar rows, which lie poleward of N96's) take the edge row's
  values. On regional source grids the same applies beyond the edge columns.
  Cyclic grids interpolate across the wrap.
- **Rotated source grids**: interpolation happens in the source grid's own
  (rotated) frame.
- **Ties at coastal points**: when several valid corners share the largest weight,
  the first in the order south-west, south-east, north-west, north-east wins.
- **Ties at unresolved points**: when several resolved points are equally near,
  the one nearer the equator (in the target grid's own frame) wins, then the
  western one.
- **No resolved points at all**: if no target point has any valid corner, each
  takes its nearest valid source point. This is a generic fallback the UM never
  meets for land fields.
- **Summation order**: interpolated values are summed in corner order, which keeps
  results bitwise identical to the UM's.

#### How the rules were established

The rules were reverse-engineered from native reconfigurations of an ACCESS-AM3
N96 dump, `am3a.ainitial_n96e`. These were run with `h_int_method=1` and
`coast_adj_method=3`, onto N48 (`am3a.astart_n48e`, grid points aligned with
N96) and N512 (`am3a.astart_n512e`, not aligned). The comparisons loaded the
dumps with `iris.load`, used the land-sea mask as the mask, and went through the
cube interface. The dumps are several GB, so these comparisons are not part of
the test suite; the tests check the same rules on small synthetic grids.

| Field | N48 | N512 |
| --- | --- | --- |
| 22 canopy water, 213 canopy conductance | Bitwise identical | Identical except 13 and 16 points, all exact coastal weight ties |
| 229 canopy water on tiles (17 tiles) | Bitwise identical | Identical except 22 point-tiles, all exact ties |
| 20 soil temperature, after the lapse-rate adjustment below | Identical (to 1e-10 K) | Identical except 18 points, all exact ties |
| 9 soil moisture, after the stress conversion below | 95–96% of points per layer | 95–96% of points per layer |

In the 16 N512 coastal ties, the UM breaks the tie in a way no fixed corner order
or floating-point rounding reproduces. These are the only known differences in
interpolation.

#### Field-specific adjustments the UM makes afterwards

These are not done by this library. A script that wants to match the UM must
apply them:

- **Soil temperature (stash 20)**: subtract 6 K per km of orography increase,
  where the increase is target orography minus source orography interpolated with
  *unmasked* bilinear interpolation (all-true mask cubes). Requires
  `l_adj_t_soil=.true.`.
- **Soil moisture (stash 9)**: interpolate the stress
  `(θ − θ_wilt) / (θ_crit − θ_wilt)`, computed with the source wilting (40) and
  critical (41) points, then convert back with the target's (`use_smc_stress`).
  The remaining 4–5% of points are mostly next to source land-ice points, where
  `θ_wilt = θ_crit`, and are not yet explained.
- **Tile surface temperature (stash 233)**: further tile-specific processing that
  has not been worked out.

#### Unverified

- Target points that lie exactly on a source row or column: all four corners are
  checked against the mask, including those with zero weight.
- Regional source grids.
- Rotated-pole source and target grids.

The N48 and N512 checks used only global, unrotated grids that never align
exactly.

### Edge cases

- **Empty source mask, non-empty target mask**: `build_regridder` raises
  `EmptySourceMaskError` (a `ValueError`). There is nothing to project from, and
  what to do instead is a decision for the caller.
- **Empty target mask**: valid; the output is entirely masked (or `fill_value`).
- **Missing data inside the source mask**: the regridder raises if a value
  inside the source mask is masked, non-finite or equal to
  `source_missing_value`, because the mask then does not describe where the
  field is defined. Masked or missing values outside the mask are ignored.
- **Masks must be boolean**: derive them explicitly, e.g. `tile_fraction > 0`.
  Do not infer them from field values or data masks; UM dumps can hold `0.0`
  rather than missing data on inactive tiles.
- **Lazy data** is realised when regridded.
- **Denormals**: the UM flushes denormal numbers to zero and numpy does not, so
  values below about 1e-300 may differ from the UM's zeros.

### How far values are borrowed from

Every valid target point is filled, however far away the nearest valid point is.
`nearest_source_distances(regridder)` returns a cube on the target grid holding
the distance from each valid target point to its nearest valid source point, in
radians (multiply by 6371 for kilometres), masked outside the target mask. A
caller can use it to flag points whose values came from far away. This matters
when the source and target describe the land surface differently, since a tile
may exist at many target points with no source point of that tile anywhere
nearby.

### Comparison with ANTS

ANTS (2.2.0) can regrid with `cube.regrid(target, ants.regrid.GeneralRegridScheme(horizontal_scheme="Linear"))`
and then fill missing land points with `ants.analysis.make_consistent_with_lsm`.
That differs from the UM reconfiguration:

- **Coastal points**: the Linear regrid masks every target point with any masked
  corner. `make_consistent_with_lsm` then fills it by spiral search from the
  nearest unmasked point *of the regridded field*, i.e. an interior target point,
  rather than copying the heaviest valid source corner.
- **Unresolved points**: likewise filled from interior target points only, where
  the UM also uses coastal ones.
- **Edge rows**: Linear returns NaN beyond the first and last source rows by
  default, which is then filled by search rather than taking the edge row values.
- **Masks**: ANTS only knows the source mask through masked data, so tile fields
  must be masked explicitly before regridding.

For N96 → N48 canopy water (stash 22) with the land-sea mask, ANTS differs from
the native UM at 432 of 566 coastal points (by up to the full range of the
field) and 42 of 68 unresolved points. `um_masked_regrid` is bitwise identical
at all of them.
