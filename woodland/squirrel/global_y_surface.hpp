#ifndef INCLUDE_WOODLAND_SQUIRREL_GLOBAL_Y_SURFACE
#define INCLUDE_WOODLAND_SQUIRREL_GLOBAL_Y_SURFACE

#include "woodland/squirrel/mesh_based_surface.hpp"

namespace woodland {
namespace squirrel {

// Exact parameterized surface p(x,z) = (x, y(x,z), z). The LCS uses a
// +y-oriented normal and global +x as its primary tangent direction.
struct GlobalYSurface : public MeshBasedSurface {
  struct UserFn {
    typedef std::shared_ptr<UserFn> Ptr;
    typedef std::shared_ptr<const UserFn> CPtr;
    // g = (partial y / partial x, partial y / partial z); g can be nullptr.
    virtual void eval(const Real x, const Real z, Real& y, RPtr g) const = 0;
  };

  GlobalYSurface(const mesh::Mesh::CPtr& m, const UserFn::CPtr& ufn,
                 const bool support_req0=true, const bool mesh_scale=true);

  const char* name () const override { return "GlobalYSurface"; }

  void cell_ctr_xyz(const Idx ci, Real xyz[3]) const override;
  void cell_ctr_lcs(const Idx ci, Real lcs[9]) const override;
  bool can_interpolate_outside_cell () const override { return not scale(); }
  bool supports_J() const override;
  bool calc_cell_lcs(const Idx ci, const Real xyz[3], Real uv[2]) const override;

  // uv is (x,z). If scale(ci), uv on input is in the scaled coordinates.
  void cell_position(const Idx ci, const int n, CRPtr uv, RPtr xyz,
                     RPtr lcs = nullptr, RPtr jacdet = nullptr,
                     RPtr jac = nullptr) const override;

private:
  UserFn::CPtr ufn;
  const bool support_req0;
  RealArray y_ctrs, lcs_ctrs;

  void init();
};

} // namespace squirrel
} // namespace woodland

#endif
