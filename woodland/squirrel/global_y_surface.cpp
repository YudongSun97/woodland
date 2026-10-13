#include "woodland/squirrel/global_y_surface.hpp"

#include "woodland/acorn/util.hpp"
#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/dbg.hpp"

namespace woodland {
namespace squirrel {

typedef acorn::Matvec<2,Real> mv2;
typedef acorn::Matvec<3,Real> mv3;
using mesh::Mesh;

static void asserti_y (const mesh::Mesh& m, const Idx ci) {
  assert(ci >= 0 && ci < m.get_ncell());
}

GlobalYSurface
::GlobalYSurface (const Mesh::CPtr& m, const UserFn::CPtr& ufn_,
                  const bool support_req0_, const bool mesh_scale)
  : MeshBasedSurface(m, mesh_scale, 0, 2),
    ufn(ufn_), support_req0(support_req0_)
{ init(); }

void GlobalYSurface::init () {
  const auto nc = m->get_ncell();
  y_ctrs.resize(nc);
  lcs_ctrs.resize(9*nc);
  ompparfor for (Idx ci = 0; ci < nc; ++ci) {
    Real xyz[3];
    const auto lcs = &lcs_ctrs[9*ci];
    lcs[0] = 1; lcs[1] = 0; lcs[2] = 0;
    cell_position1(ci, &uv_ctrs[2*ci], xyz, lcs);
    y_ctrs[ci] = xyz[1];
    assert(not std::isnan(y_ctrs[ci]));
  }
}

void GlobalYSurface::cell_ctr_xyz (const Idx ci, Real xyz[3]) const {
  asserti_y(*m, ci);
  xyz[0] = uv_ctrs[2*ci  ];
  xyz[1] = y_ctrs[ci];
  xyz[2] = uv_ctrs[2*ci+1];
}

void GlobalYSurface::cell_ctr_lcs (const Idx ci, Real lcs[9]) const {
  asserti_y(*m, ci);
  acorn::copy(9, &lcs_ctrs[9*ci], lcs);
}

bool GlobalYSurface
::calc_cell_lcs (const Idx ci, const Real xyz[3], Real uv[2]) const {
  uv[0] = xyz[0];
  uv[1] = xyz[2];
  apply_scale(ci, true, uv);
  return true;
}

bool GlobalYSurface::supports_J () const { return support_req0; }

void GlobalYSurface
::cell_position (const Idx ci, const int n, CRPtr uv, RPtr xyz_, RPtr lcs_,
                 RPtr jacdet, RPtr jac) const {
  const bool scaled = scale(ci);
  for (int i = 0; i < n; ++i) {
    Real xz[2] = {uv[2*i], uv[2*i+1]};
    if (scaled) apply_scale(ci, false, xz);
    Real y;
    ufn->eval(xz[0], xz[1], y, nullptr);
    auto xyz = &xyz_[3*i];
    xyz[0] = xz[0]; xyz[1] = y; xyz[2] = xz[1];

    Real grad[2];
    if (jacdet or jac or lcs_)
      ufn->eval(xz[0], xz[1], y, grad);

    if (jacdet) {
      jacdet[i] = std::sqrt(1 + mv2::norm22(grad));
      if (scaled) jacdet[i] *= get_jacdet_scale(ci);
    }
    if (jac) {
      auto J = &jac[6*i];
      if (scaled) {
        Real A[4];
        get_jacobian_scale(ci, A);
        // [1 0; y_x y_z; 0 1] A, stored row major as 3x2.
        J[0] = A[0]; J[1] = A[2];
        J[2] = grad[0]*A[0] + grad[1]*A[1];
        J[3] = grad[0]*A[2] + grad[1]*A[3];
        J[4] = A[1]; J[5] = A[3];
      } else {
        J[0] = 1;       J[1] = 0;
        J[2] = grad[0]; J[3] = grad[1];
        J[4] = 0;       J[5] = 1;
      }
    }
    if (lcs_) {
      // +y-oriented normal. For y=0 this gives xhat=(1,0,0),
      // yhat=(0,0,-1), zhat=(0,1,0).
      Real zhat[] = {-grad[0], 1, -grad[1]};
      mv3::normalize(zhat);
      Real xhat[3], yhat[3];
      init_xhat_from_primary(zhat, &lcs_ctrs[9*ci], xhat);
      mv3::cross(zhat, xhat, yhat);
      mv3::normalize(yhat);
      auto lcs = &lcs_[9*i];
      mv3::copy(xhat, lcs);
      mv3::copy(yhat, &lcs[3]);
      mv3::copy(zhat, &lcs[6]);
    }
  }
}

} // namespace squirrel
} // namespace woodland
