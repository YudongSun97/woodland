#ifndef INCLUDE_WOODLAND_SQUIRREL_BICUBIC_BEZIER_SURFACE
#define INCLUDE_WOODLAND_SQUIRREL_BICUBIC_BEZIER_SURFACE

#include "woodland/squirrel/bicubic_bezier.hpp"
#include "woodland/squirrel/gallery.hpp"
#include "woodland/squirrel/mesh_based_surface.hpp"

namespace woodland {
namespace squirrel {

// C1 tensor-product bicubic Bezier reconstruction of z=f(x,y) on a
// conforming quadrilateral mesh. Each patch is constructed from analytic
// nodal f, fx, fy, and fxy data supplied by gallery::ZxyFn.
struct BicubicBezierSurface : public MeshBasedSurface {
  BicubicBezierSurface(const mesh::Mesh::CPtr& m,
                       const gallery::ZxyFn::Shape shape,
                       const Real lx = 1, const Real ly = 1);

  const char* name () const override { return "BicubicBezierSurface"; }

  void cell_ctr_xyz(const Idx ci, Real xyz[3]) const override;
  void cell_ctr_lcs(const Idx ci, Real lcs[9]) const override;

  bool can_interpolate_outside_cell () const override { return true; }
  bool supports_J () const override { return true; }
  bool calc_cell_lcs(const Idx ci, const Real xyz[3], Real uv[2]) const override;

  void cell_position(const Idx ci, const int n, CRPtr uv, RPtr xyz,
                     RPtr lcs = nullptr, RPtr jacdet = nullptr,
                     RPtr jac = nullptr) const override;

  // Packed as ncell consecutive 4x4 control grids, x/u index fastest.
  const RealArray& get_control_values () const { return ctrl; }

  static int unittest();

private:
  struct Bounds { Real x0, x1, y0, y1; };
  gallery::ZxyFn::Shape shape;
  Real lx, ly;
  std::vector<Bounds> bounds;
  RealArray ctrl, z_ctrs, lcs_ctrs;

  void init();
};

} // namespace squirrel
} // namespace woodland

#endif
