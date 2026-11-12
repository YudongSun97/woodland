## Marmot

`woodland::marmot` builds the Displacement Discontinuity Method (DDM) Green's
function matrix for a fault of arbitrary 3D shape. It follows
`woodland::squirrel`'s bicubic Bezier path (`BicubicBezierSurface`,
`ct_matrix_zxy srfrecon=1`), but the fault does not have to be a graph
`z = f(x,y)`. It can be vertical, overturned, or curved in every direction.

### Method

1. **Geometry** (`GridGeometry`). You give the fault as a structured
   `ni x nj` grid of 3D points `P(i,j)`. Marmot interpolates the grid with a C2
   tensor-product cubic spline that uses not-a-knot end conditions.
   The spline uses chord-length parameters `(s,t)`, which are roughly arc
   length along the `i` and `j` grid lines.
2. **Curved rectangular mesh** (`BezierPatchSurface`). The mesh is a
   rectangular grid in `(s,t)`. By default it is the input grid itself. With
   `nx=`, `ny=` it is uniform in `(s,t)`. Each cell is a vector bicubic Bezier
   patch built from the spline's Hermite data `(X, X_s, X_t, X_st)` at the
   cell's four corners. Adjacent patches share an edge's data, so the surface
   is C1.
3. **Matrix** (`MatrixEvaluator`, `build_matrix`). Marmot calls
   `squirrel::Stress::calc_matrix_entries`, which uses the same quadrature
   options as squirrel. The receiver is the center of a cell. Self terms use
   the Hadamard finite part, and neighbor terms use tensor quadrature.

Local frame of each cell: `zhat = X_s x X_t / |X_s x X_t|`, so the grid
ordering sets the sign of the normal. `xhat` is the tangent-plane projection of
a primary direction, and `yhat = zhat x xhat`. The `lcs=` option chooses the
primary direction:
- `strike` (default): `ez x zhat`, which is horizontal along strike. `yhat` is
  then up dip. If `zhat` points up into the hanging wall, this is the
  Aki-Richards strike.
- `x`: global x, as in squirrel. On `z = f(x,y)` the matrices then agree with
  squirrel's `srfrecon=1` matrices, which the unit test checks.
- `s`: the `s` tangent `X_s`.

Matrix entry `(ircv, isrc)` for dislocation component `d` is the stress at the
center of cell `ircv`, expressed in that cell's local frame. The dislocation's
component `d` is 1 at the center of cell `isrc` and 0 at all other cell
centers. Within each cell, the dislocation is reconstructed from the
neighboring cell values (`dislocorder`, quadratic by default). The dislocation
components are 0 (along `xhat`), 1 (along `yhat`) and 2 (opening).

### Input geometry file

`#` starts a comment. The first two numbers are `ni nj`. They are followed by
`ni*nj` lines `x y z`, with `i` varying fastest:
```
# ni nj
33 17
x y z      # P(0,0)
x y z      # P(1,0)
...
```
Example: write a grid for a curved listric fault, whose strike trace bends and
whose dip goes from 70 to 35 degrees:
```
./woodland_marmot_unittest example_geom 'n=33:17,out=geom.txt'
```

Rough faults: `tools/rough_fault_geom.py` writes a geometry file whose
roughness `h(z,x)` is a self-affine random field with two Hurst exponents. It
uses the spectral model of `rough3D_fft_minlambda_twoH.m`. It needs only
numpy. `h` can also be read from a file, for example one written by MATLAB
with `writematrix(h,'h.csv')`. `--frame yxz` (the default) gives a vertical
fault `y = y0 + h(x,z)` with normal +y; use `lcs=x` with it for squirrel's
yxz frame. `--frame dip --strike --dip --ztop` gives a dipping plane, with the
grid ordered so that marmot's normal points into the hanging wall and
`lcs=strike` gives the Aki-Richards frame. See `--help`.
```
python3 tools/rough_fault_geom.py --Hx 0.8 --Hz 0.8 --Nx 129 --Nz 65 --Lx 2 --Lz 1 \
    --alpha 0.01 --lambda-min-x 0.1 --lambda-min-z 0.1 --seed 1 --ztop 0.05 -o geom.txt
```

### Build the matrix

```
OMP_NUM_THREADS=4 ./woodland_marmot_unittest matrix 'geom=geom.txt,nx=16,ny=8,halfspace=1,out=G'
```
Options (`key=value`, separated by commas):
```
geom=FILE       fault geometry (required)
nx=, ny=        mesh cells along s (i) and t (j); default: the geometry grid
lcs=            strike | x | s [strike]
lam=, mu=       Lame parameters [1, 1]
halfspace=      0 | 1 [0]; 1 requires the fault in z <= 0
dislocorder=    0 | 1 | 2 | 3 [2]
disloc=         dislocation components, digits from 012 [012]
comps=          traction (xz, yz, zz) | all (xx, xy, xz, yy, yz, zz) [traction]
out=PREFIX      writes PREFIX.py and PREFIX_G.bin [marmot_G]
```
`mesh` takes the same options. It writes only `PREFIX.py`, so you can check
the mesh before computing the matrix.

Output:
- `PREFIX_G.bin`: float64 in native byte order, with shape
  `(n_disloc, n_comp, ncell, ncell)`. The axes are dislocation component,
  stress component, receiver and source. Cell `j*nx + i` is the cell `i`
  along `s` and `j` along `t`.
- `PREFIX.py`: the mesh (`d['m']['vtx']`, `d['m']['topo']`), cell centers
  (`d['ctr']`), local frames (`d['lcs']`, as `xhat, yhat, zhat`), cell areas,
  Bezier control points, options, and `load_G()`:
```python
import sys; sys.path.insert(0, 'dir/of/output')
from G import d, load_G
G = load_G()                 # G[di, ci, rcv, src]
tau_strike = G[0, 0] @ slip  # xz traction from slip along xhat
```

Cost grows as `ncell^2`. Each entry needs up to about ten curved-element
integrals, and twice that in a half space. On 4 cores, the 128-cell example
with 3 dislocation components in a half space takes about 70 s. For large
meshes, use the C API (`marmot_capi.h`, `woodland_marmot_entries`) to compute
entries on demand, for example with an H-matrix code.

### Unit tests

`./woodland_marmot_unittest` (or `ctest`) checks the following:
- Spline polynomial reproduction, exactness on planes, and 4th-order
  convergence.
- C1 continuity of the patches, area, normals and frames on a quarter
  cylinder (not a graph), and the closest-point inverse.
- Agreement with squirrel's bicubic `ct_matrix_zxy` matrices on `z = f(x,y)`,
  in the whole space and the half space, to rounding. The hypersingular self
  terms amplify rounding to about 1e-10 relative.
- Rotation and translation invariance of the matrices for a general surface.
- The example listric fault in a half space, including the output files.
