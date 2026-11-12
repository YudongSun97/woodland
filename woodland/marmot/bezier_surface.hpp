#ifndef INCLUDE_WOODLAND_MARMOT_BEZIER_SURFACE
#define INCLUDE_WOODLAND_MARMOT_BEZIER_SURFACE

#include "woodland/marmot/geometry.hpp"
#include "woodland/squirrel/discretization.hpp"

namespace woodland {
namespace marmot {

// C1 piecewise vector bicubic Bezier reconstruction of a Geometry on a
// rectangular grid in its (s,t) parameter space: the curved rectangular mesh
// has nodes X(sb[i], tb[j]), and cell (i,j), with index j*nx + i, is the
// bicubic Bezier patch with the geometry's nodal Hermite data (X, X_s, X_t,
// X_st) at its four corners. Adjacent patches share an edge's Hermite data, so
// the reconstruction is C1 in (s,t).
//   This generalizes squirrel::BicubicBezierSurface from z = f(x,y) with
// analytic derivatives to arbitrary 3D surfaces with derivatives from the
// Geometry. The cell (u,v) coordinates are the global (s,t), the cell polygons
// are the axis-aligned rectangles [sb[i], sb[i+1]] x [tb[j], tb[j+1]], and the
// Mesh's vertices are the 3D node positions.
//   Local coordinate system (LCS): zhat = X_s x X_t / |X_s x X_t|, so the
// node ordering sets the normal's sign. xhat is the projection onto the tangent
// plane of a per-cell primary direction, yhat = zhat x xhat.
struct BezierPatchSurface : public squirrel::Surface {
  typedef std::shared_ptr<BezierPatchSurface> Ptr;
  typedef std::shared_ptr<const BezierPatchSurface> CPtr;

  // Primary direction of xhat:
  //   strike: ez x zhat, horizontal and along strike. Then yhat is the
  //           projection of ez onto the tangent plane: up dip.
  //   x:      global x, as in squirrel; matches BicubicBezierSurface.
  //   s:      X_s at the cell center.
  // If the primary direction is (nearly) normal to the surface, s is used.
  enum class LcsPrimary { strike, x, s };
  static LcsPrimary convert(const std::string& s);
  static std::string convert(const LcsPrimary p);

  // sb has nx+1 and tb ny+1 increasing breakpoints.
  BezierPatchSurface(const Geometry& g, const RealArray& sb,
                     const RealArray& tb,
                     const LcsPrimary primary = LcsPrimary::strike);

  const char* name () const override { return "BezierPatchSurface"; }

  const squirrel::mesh::Mesh::CPtr& get_mesh () const { return m; }
  int get_nx () const { return nx; }
  int get_ny () const { return ny; }
  const RealArray& get_sb () const { return sb; }
  const RealArray& get_tb () const { return tb; }
  LcsPrimary get_lcs_primary () const { return primary; }

  Idx  get_ncell () const override { return Idx(nx)*ny; }
  int  cell_nvtx (const Idx) const override { return 4; }
  void cell_vtxs_uv(const Idx ci, RPtr uv) const override;
  void cell_ctr_uv (const Idx ci, Real uv [2]) const override;
  void cell_ctr_xyz(const Idx ci, Real xyz[3]) const override;
  void cell_ctr_lcs(const Idx ci, Real lcs[9]) const override;

  bool supports_J () const override { return true; }
  bool can_interpolate_outside_cell () const override { return true; }
  // Closest point to xyz on the (extended) patch of cell ci, by Gauss-Newton.
  bool calc_cell_lcs(const Idx ci, const Real xyz[3], Real uv[2])
    const override;

  void cell_position(const Idx ci, const int n, CRPtr uv, RPtr xyz,
                     RPtr lcs = nullptr, RPtr jacdet = nullptr,
                     RPtr jac = nullptr) const override;

  // Bezier control points, ncell x 3 (x,y,z) x 16, within a 4x4 grid the u (s)
  // index fastest.
  const RealArray& get_control_points () const { return ctrl; }

  static int unittest();

private:
  int nx, ny;
  RealArray sb, tb;
  LcsPrimary primary;
  squirrel::mesh::Mesh::CPtr m;
  RealArray ctrl, xyz_ctrs, lcs_ctrs, primaries;

  void init(const Geometry& g);
  void cell_ij (const Idx ci, int& i, int& j) const { i = ci % nx; j = ci / nx; }
};

// Uniform breakpoints: n+1 points from a to b.
RealArray uniform_breaks(const Real a, const Real b, const int n);

} // namespace marmot
} // namespace woodland

#endif
