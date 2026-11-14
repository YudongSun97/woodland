# Woodland on flat triangles with a continuous slip direction (nodal normals)

This branch holds a small modification of Woodland and the scripts of the test we ran with it. The test belongs to
our work on the kink error of faceted faults (paper in preparation).

On flat triangles, Woodland (`flatelem`) gives the slip components in the local coordinate system of each facet. The
slip direction therefore follows the facet tangents and jumps at every element edge. For uniform slip this is exactly
the constant-slip triangular dislocation element (HBI, Nikkhoo & Walter 2015): the two codes agree to 2×10⁻⁸ or better
on our meshes. On a curved fault the normal stress of this element does not converge under refinement.

The modification adds a second local coordinate system, the **slip LCS**, built from a normal field that is continuous
across the elements (nodal normals):

- the slip components are given in the slip LCS, and the receiver stress is resolved in it;
- the kernel and the integration geometry are unchanged: the integration surface is still the flat triangle, with its
  facet normal.

The slip vector is then continuous across the element edges, and the normal stress converges:

- at first order with linearly blended vertex normals;
- at second order with a quadratic blend of exact normals;
- to errors of 10⁻³–10⁻⁵ with the exact normal at every quadrature point.

For **non-uniform slip**, the flat facets limit every variant to first order, even with the exact slip magnitude and
direction at every quadrature point. The cause is the offset of the facets from the fault (see *Interpretation*).

## Base version and license

- This branch starts from commit 4dafa06 ("include algotithm", 2026-10-13) of the `myfeature` branch, whose tree is
  the Woodland tree of FDRA2C3D-woodland (`work/woodland`) that the modification was made on.
- The source changes are the 9 files under `woodland/squirrel` listed below; `nodal_normal/` holds this README, the
  test scripts and the change as a patch (`woodland_nodal_normal.patch`, `patch -p1` from the repository root).
- Woodland is by Andrew M. Bradley (https://github.com/ambrad/woodland, BSD 3-Clause License, see `LICENSE` and
  `AUTHORS`).

## Contents

```
woodland/squirrel/             the 9 changed or new files (see Changes)
nodal_normal/
  README.md                    this file
  woodland_nodal_normal.patch  the modification as a unified diff against 4dafa06
  tests/
    smallslope.py              geometry of the test, vertex normals (Max, quadric fit, exact), fault triangulation,
                               first-order small-slope reference for uniform slip
    smallslope_var.py          the reference for the non-uniform slip s(x, y)
    run_nn.py                  uniform slip, all variants (FA, NN, QF, EX, EX6, EXP)
    run_var.py                 non-uniform slip, with the slip-reconstruction order 0-3 or the exact slip (oracle)
    check_geom_term.py         the facet-offset term for non-uniform slip, against the error of the EXP oracle
```

## Changes

Every new hook has a default that keeps the previous behaviour, so the other `Surface` classes and the existing tests
are unchanged.

### `discretization.hpp`: three virtual hooks on `Surface`

```cpp
// Slip LCS at n points uv of cell ci; false: use the geometric LCS (default).
virtual bool cell_slip_lcs(const Idx ci, const int n, CRPtr uv, RPtr lcs) const;
// Slip LCS at the cell centre (default: cell_ctr_lcs).
virtual void cell_ctr_slip_lcs(const Idx ci, Real lcs[9]) const;
// Oracle: the slip magnitude (first component) at n global points, replacing the reconstruction; false (default)
// if not provided.
virtual bool slip_magnitude(const int n, CRPtr xyz, RPtr s) const;
```

### `calc_stress.cpp`

- **`Integrands::eval`:**
  - The workspace per point grows from 16 to 26 values: the slip LCS (9) and the slip magnitude (1).
  - The reconstructed dislocation components are rotated to global coordinates with the slip LCS when the surface
    provides one, and with the geometric LCS otherwise.
  - The kernel keeps the geometric LCS (`&lcs[9*k+6]`, the normal of the integration surface).
  - If `slip_magnitude` is provided, the components at the point are replaced by (s, 0, 0).
- **`calc_matrix_entries`:** the receiver stress is rotated with `cell_ctr_slip_lcs`.

### `ctzx_eval.cpp`

The receiver rotation uses `cell_ctr_slip_lcs`.

### `ctzx_surface.hpp/.cpp`, `FlatElementSurface`

Four normal modes:

| mode | how it is set | normal field |
|---|---|---|
| `mode_facet` (default) | — | facet normal; Woodland as before |
| `mode_vertex` | `set_vertex_normals(vn)` | barycentric blend of vertex normals (nvtx × 3), normalized |
| `mode_quad` | `set_element_normals6(v)` | quadratic Lagrange blend of six normals per triangle (vertices 0, 1, 2, then the midpoints of edges 01, 12, 20; nt × 6 × 3), normalized |
| `mode_fn` | `set_normal_fn(f)` | a normal function of the point |

- **Building the slip LCS (`slip_lcs_at`):**
  - z-hat is the normal field at the point;
  - x-hat is the primary direction projected onto the tangent plane (`init_xhat_from_primary`, the convention of the
    geometric LCS);
  - y-hat = z-hat × x-hat.
- **Oracle slip:** `set_slip_fn(f)` sets the exact slip magnitude as a function of the point.
- **Primary direction:** `init` now stores it (`primary_`).

### `nn_flat.cpp` (new) and its registration

`testmain.cpp`, `ctzx.hpp` and `CMakeLists.txt` register the new command:

```
woodland_squirrel_unittest nn_flat in=IN.bin,out=OUT.bin
```

It reads a triangulation, puts a dislocation of magnitude s along the local x direction of the slip LCS on every
triangle, and writes the stress at the centroids, resolved in the slip LCS of the receiver. The N² interactions are
summed directly with `Stress::calc_s1_r1`, with the standard options for neighbours and non-neighbours.

**Input** (little endian):

```
int32  nv, nt, mode, disloc_order, hill, has_slip
double A, lambda, mu
double V[nv][3]; int32 T[nt][3]
double N[nv][3]       if mode == 1   vertex normals
double N6[nt][6][3]   if mode == 3   six normals per triangle
double s[nt]          if has_slip    slip magnitude at the centroids (else 1), reconstructed with disloc_order
```

**Mode:**

| mode | normal field |
|---|---|
| 0 | facet normal |
| 1 | vertex normals, linear blend |
| 2 | exact normal of the test hill at every point (hill 0: `md`, 1: `c3`, zero outside [0,1]²) |
| 3 | six normals per triangle, quadratic blend |

`has_slip = 2` also turns on the oracle: the exact s(x, y) = 1 + ½cos(2πx)cos(2πy) at every quadrature point.

**Output:** `double[nt][9]`: the centroid (3), then the stress (xx, xy, xz, yy, yz, zz) in the slip LCS of the
receiver. `zz` is σ_nn along the normal of the slip LCS at the centroid.

Mode 2 and the oracle slip are hard-coded for our test (`hill_fg`, the lambda in `run_nn_flat`). For other surfaces,
pass the normals with mode 1 or 3, or change the two lambdas.

## Build

Build as for Woodland, from the root of the repository.

On macOS (M3 Pro) we used the system clang with Homebrew libomp, because gcc-15 there finds no libstdc++ headers:

```bash
mkdir build && cd build
CC=/usr/bin/clang CXX=/usr/bin/clang++ FC=gfortran-15 cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
make -j4
```

## The test

| | |
|---|---|
| Fault | z = h(x, y) over S = [−0.25, 1.25]²; hill on D = [0, 1]², flat margin around it |
| Hill (`c3`) | h = A [cos 2πx − cos 4πx] sin²(πx) [1 − cos 2πy]² on D, h = 0 outside; the first three derivatives vanish on ∂D, so the curvature is continuous |
| A | 10⁻⁴ (small-slope regime) |
| Medium | full space, λ = μ = 1 |
| Mesh | structured: squares of edge h = 1/8, 1/16, 1/32, 1/64 (levels 0–3; 288 to 18 432 triangles), each cut along the diagonal (0,0)–(1,1), vertices on the surface |
| Uniform slip | unit slip along the local x direction (e_x projected onto the tangent plane) |
| Non-uniform slip | s = 1 + ½cos(2πx)cos(2πy) along the same direction; s = 1 on ∂S |
| Reference | first-order small-slope solution σ_nn = A σ₁(x, y) + O(A³), see below |
| Error | relative L2 error of σ_nn at the centroids, area-weighted, over the whole fault |

**Reference:**

- **Uniform slip:** σ₁ is a principal-value integral of the curvature plus closed-form terms (`smallslope.sigma1`),
  accurate to about 10⁻¹¹.
- **Non-uniform slip:** A σ₁ˢ = s(x) A σ₁ + C ∫ [s(ξ) − s(x)] K/R⁵, with K the first-order kernel, which is O(R³) at
  the receiver (`smallslope_var.correction`). It agrees with a direct principal value to 2×10⁻⁸ of the maximum inside
  the hill.
- **Model error:** about 3×10⁻⁵ for A = 10⁻⁴. It matters only for EXP at h = 1/64 (see below).

**Variants** (`run_nn.py`; the paper names in parentheses):

| variant | slip direction | continuous across edges |
|---|---|---|
| FA | facet tangent (Woodland as it is) | no |
| NN | linear blend of Max (1999) vertex normals | yes |
| QF | linear blend of quadric-fit vertex normals | yes |
| EX | linear blend of exact vertex normals | yes |
| EX6 (EXN) | quadratic blend of exact normals at the vertices and the edge midpoints | yes |
| EXP (EXQ) | exact normal at every quadrature point (oracle) | yes |

Run from `nodal_normal/tests/` (Python 3 with numpy; `check_geom_term.py` also needs matplotlib):

```bash
export WOODLAND_BIN=/path/to/build/woodland/squirrel/woodland_squirrel_unittest OMP_NUM_THREADS=4
python3 run_nn.py 0                  # uniform slip, level 0, all variants
python3 run_var.py 0                 # non-uniform slip, level 0
python3 check_geom_term.py 0,1 2     # facet-offset term against the error of EXP:x (after the two runs above)
```

Results go to `tests/results/`. Level 0 takes seconds. Level 3 (18 432 elements) takes 340–700 s per variant on
3 threads of an M3 Pro.

## Results

The uniform-slip numbers are from the full runs (levels 0–3). We checked the scripts of this branch at level 0, where they
reproduce them to all digits.

### Uniform slip: relative L2 error of σ_nn

| variant | h = 1/8 | 1/16 | 1/32 | 1/64 | orders |
|---|---|---|---|---|---|
| FA | 0.668 | 0.765 | 0.818 | 0.836 | does not converge |
| NN | 0.772 | 0.382 | 0.158 | 0.0659 | 1.01, 1.28, 1.26 |
| QF | 0.946 | 0.582 | 0.239 | 0.0895 | 0.70, 1.29, 1.41 |
| EX | 0.600 | 0.273 | 0.121 | 0.0559 | 1.14, 1.17, 1.12 |
| EX6 | 0.122 | 0.0332 | 0.00861 | 0.00218 | 1.88, 1.95, 1.98 |
| EXP | 1.05×10⁻³ | 2.96×10⁻⁴ | 8.15×10⁻⁵ | 3.29×10⁻⁵ | 1.83, 1.86, 1.31 |
| HBI, constant slip tilted to the exact normal at the centroid | 0.146 | 0.103 | 0.096 | 0.094 | stalls near 10% |

- **Woodland FA against HBI:** they differ by 2.7×10⁻¹⁰, 1.3×10⁻⁹, 5.3×10⁻⁹ and 2.3×10⁻⁸ (relative, h = 1/8 to 1/64).
- **EXP at h = 1/64:** the order drops to 1.31 because of the model error of the reference, not the discretization.
  Halving A to 5×10⁻⁵ and then to 2.5×10⁻⁵ gives 2.05×10⁻⁵ and 1.89×10⁻⁵ at h = 1/64. At h = 1/32 the error only
  changes from 8.15×10⁻⁵ to 7.53×10⁻⁵ (A = 5×10⁻⁵). Without the model error, EXP converges at order 2.0.
- **Reconstruction order:** for uniform slip the components are constant, so the reconstruction is exact. Orders 0 and
  2 give identical stresses (checked at h = 1/16).

### Non-uniform slip: slip-reconstruction order 2

| variant | h = 1/8 | 1/16 | 1/32 | 1/64 | orders |
|---|---|---|---|---|---|
| FA | 0.689 | 0.808 | 0.881 | 0.910 | does not converge |
| NN | 0.757 | 0.391 | 0.168 | 0.0724 | 0.95, 1.22, 1.21 |
| QF | 0.900 | 0.569 | 0.242 | 0.0942 | 0.66, 1.23, 1.36 |
| EX | 0.616 | 0.294 | 0.134 | 0.0632 | 1.07, 1.13, 1.09 |
| EX6 | 0.149 | 0.0443 | 0.0146 | 0.00577 | 1.75, 1.61, 1.33 |
| EXP | 0.0586 | 0.0242 | 0.0110 | 0.00525 | 1.27, 1.14, 1.07 |
| EXP with the exact s at every point (`EXP:x`) | 0.0475 | 0.0225 | 0.0108 | 0.00522 | 1.08, 1.06, 1.04 |
| EXP, order 0 (constant magnitude per element) | 0.101 | 0.0544 | 0.0283 | 0.0143 | 0.89, 0.94, 0.98 |
| HBI, constant slip tilted to the exact normal | 0.198 | 0.122 | 0.110 | 0.107 | stalls near 11% |

- **Reconstruction order:** orders 1 and 3 are within 2% of the oracle at h = 1/64.
- **Slip amplitude:** the error is proportional to the amplitude of the variation of s. Halving it from 0.5 to 0.25
  halves the error.

## Interpretation

- **Uniform slip:** expanding the dislocation kernel to first order in the slope, the facet geometry cancels exactly.
  - A dislocation with a smooth slip direction gives the same first-order stress on the fault and on the polyhedron
    through its vertices.
  - What remains is the spurious tensile component ε of the slip direction. It acts through the stiffness |k| ~ 1/h,
    so Δσ = O(h^(q−1)) for ε = O(h^q).
  - The orders follow: linear blend q = 2, first order; quadratic blend of exact normals q = 3, second order;
    FA and constant slip have a jump in ε at every edge and do not converge.
- **Non-uniform slip:** the cancellation needs a constant Burgers vector. With a slip magnitude s(ξ), a term in ∇s
  remains:

  Δσ_δ = (3/2) C HFP ∫ δ X (X s_ξ + Y s_η) / R⁵ dξ dη,    C = μ / (4π(1 − ν)),

  where δ = h_I − h is the offset of the flat facets from the fault.
  - The kernel is hypersingular, and δ = O(h²κ) varies on the element scale, so Δσ_δ = O(μ |∇s| h κ). That is first
    order in h and of the same order in the slope as σ₁.
  - `check_geom_term.py` evaluates Δσ_δ by FFT on the facet offset, using the Fourier symbol
    FT[x_i x_j / |x|⁵] = −(2π/3)(k_i k_j / |k| + δ_ij |k|).
  - It correlates with the error of `EXP:x` to 0.991 at h = 1/8 and 0.998 or better on the finer meshes.
  - The difference, relative to the solution, is 2.7×10⁻², 6.9×10⁻³, 1.7×10⁻³ and 4.5×10⁻⁴, so it converges at
    second order.
- **Practical consequence:** on flat elements with nodal normals and a continuous slip direction, the normal stress
  converges at first order for general slip.
  - Beyond first order, the elements must follow the fault (curved elements), or the facet offset must be corrected.
  - The order of the slip reconstruction changes only the constant.
  - The slip magnitude enters the normal stress only through the first-order kernel, because tangential slip on a flat
    fault produces no normal stress.

## Caveats

- **Meaning of the components:** with the slip LCS, the dislocation components refer to the slip LCS, not to the facet.
  In a code that uses them as unknowns, the opening component is along the nodal normal, and the receiver tractions
  are resolved in the slip LCS.
- **What was tested:** only the direct summation of `nn_flat` (`Stress::calc_s1_r1`). The matrix-assembly path
  (`calc_matrix_entries`) uses the same integrand and the same receiver rotation, but we did not run it.
- **Surfaces:** only `FlatElementSurface` implements the new hooks. The curved surfaces keep the geometric LCS.
