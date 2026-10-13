#include "woodland/squirrel/bicubic_bezier_surface.hpp"

#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/util.hpp"

#include <algorithm>
#include <cmath>

namespace woodland {
namespace squirrel {

namespace {
using mv2 = acorn::Matvec<2,Real>;
using mv3 = acorn::Matvec<3,Real>;

void assert_ci (const mesh::Mesh& m, const Idx ci) {
  assert(ci >= 0 && ci < m.get_ncell());
}
} // namespace

BicubicBezierSurface
::BicubicBezierSurface (const mesh::Mesh::CPtr& m,
                        const gallery::ZxyFn::Shape shape_,
                        const Real lx_, const Real ly_)
  : MeshBasedSurface(m, false, 0, 1), shape(shape_), lx(lx_), ly(ly_)
{
  assert(lx > 0 && ly > 0);
  init();
}

void BicubicBezierSurface::init () {
  const auto& topo = *m->get_topo();
  const auto& vtxs = *m->get_vtxs();
  const Idx nc = m->get_ncell();
  bounds.resize(nc);
  ctrl.resize(16*nc);
  z_ctrs.resize(nc);
  lcs_ctrs.resize(9*nc);

  ompparfor for (Idx ci = 0; ci < nc; ++ci) {
    const auto cell = topo.get_cell(ci);
    assert(cell.n == 4);
    auto& b = bounds[ci];
    b.x0 = b.y0 = 1e100;
    b.x1 = b.y1 = -1e100;
    for (int k = 0; k < 4; ++k) {
      const auto p = vtxs.get_vtx(cell[k]);
      b.x0 = std::min(b.x0, p[0]); b.x1 = std::max(b.x1, p[0]);
      b.y0 = std::min(b.y0, p[1]); b.y1 = std::max(b.y1, p[1]);
    }
    const Real hx = b.x1-b.x0, hy = b.y1-b.y0;
    assert(hx > 0 && hy > 0);

    const Real xs[4] = {b.x0, b.x1, b.x0, b.x1};
    const Real ys[4] = {b.y0, b.y0, b.y1, b.y1};
    Real f[4], fx[4], fy[4], fxy[4];
    for (int k = 0; k < 4; ++k) {
      gallery::ZxyFn::eval(shape, xs[k]/lx, ys[k]/ly,
                           f[k], &fx[k], &fy[k], &fxy[k]);
      fx[k] /= lx;
      fy[k] /= ly;
      fxy[k] /= lx*ly;
    }
    BicubicBezier::init_from_hermite(
      hx, hy, f, fx, fy, fxy, &ctrl[16*ci]);

    Real xyz[3];
    auto lcs = &lcs_ctrs[9*ci];
    lcs[0] = 1; lcs[1] = 0; lcs[2] = 0;
    cell_position1(ci, &uv_ctrs[2*ci], xyz, lcs);
    z_ctrs[ci] = xyz[2];
  }
}

void BicubicBezierSurface::cell_ctr_xyz (const Idx ci, Real xyz[3]) const {
  assert_ci(*m, ci);
  xyz[0] = uv_ctrs[2*ci];
  xyz[1] = uv_ctrs[2*ci+1];
  xyz[2] = z_ctrs[ci];
}

void BicubicBezierSurface::cell_ctr_lcs (const Idx ci, Real lcs[9]) const {
  assert_ci(*m, ci);
  acorn::copy(9, &lcs_ctrs[9*ci], lcs);
}

bool BicubicBezierSurface
::calc_cell_lcs (const Idx ci, const Real xyz[3], Real uv[2]) const {
  assert_ci(*m, ci);
  uv[0] = xyz[0];
  uv[1] = xyz[1];
  return true;
}

void BicubicBezierSurface
::cell_position (const Idx ci, const int n, CRPtr uv, RPtr xyz_, RPtr lcs_,
                 RPtr jacdet, RPtr jac) const {
  assert_ci(*m, ci);
  const auto& b = bounds[ci];
  const Real hx = b.x1-b.x0, hy = b.y1-b.y0;
  for (int k = 0; k < n; ++k) {
    const Real x = uv[2*k], y = uv[2*k+1];
    const Real u = (x-b.x0)/hx, v = (y-b.y0)/hy;
    Real z, zu, zv;
    BicubicBezier::eval(&ctrl[16*ci], u, v, z, &zu, &zv);
    const Real zx = zu/hx, zy = zv/hy;
    auto xyz = &xyz_[3*k];
    xyz[0] = x; xyz[1] = y; xyz[2] = z;
    if (jacdet) jacdet[k] = std::sqrt(1 + zx*zx + zy*zy);
    if (jac) {
      auto J = &jac[6*k];
      J[0] = 1; J[1] = 0;
      J[2] = 0; J[3] = 1;
      J[4] = zx; J[5] = zy;
    }
    if (lcs_) {
      Real zhat[] = {-zx, -zy, 1};
      mv3::normalize(zhat);
      Real xhat[3], yhat[3];
      init_xhat_from_primary(zhat, &lcs_ctrs[9*ci], xhat);
      mv3::cross(zhat, xhat, yhat);
      mv3::normalize(yhat);
      auto lcs = &lcs_[9*k];
      mv3::copy(xhat, lcs);
      mv3::copy(yhat, &lcs[3]);
      mv3::copy(zhat, &lcs[6]);
    }
  }
}

int BicubicBezierSurface::unittest () {
  int nerr = 0;
  const auto shape = gallery::ZxyFn::Shape::cos10_cos12_y;
  const auto make_grid = [shape] (const int n) {
    const auto vtxs = mesh::make_vertices(3);
    const auto topo = mesh::make_topology();
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i) {
        const Real x = Real(i)/n, y = Real(j)/n;
        Real z;
        gallery::ZxyFn::eval(shape, x, y, z, nullptr, nullptr);
        const Real p[] = {x,y,z};
        vtxs->append_vtx(p);
      }
    const auto vi = [n] (const int i, const int j) { return j*(n+1)+i; };
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Idx cell[] = {vi(i,j), vi(i+1,j), vi(i+1,j+1), vi(i,j+1)};
        topo->append_cell(4, cell);
      }
    topo->end_modifications();
    return squirrel::mesh::make_mesh(vtxs, topo);
  };

  const auto mesh2 = make_grid(2);
  BicubicBezierSurface s(mesh2, shape);

  // Adjacent patches must agree in position and first derivatives all along
  // their shared edge.
  for (int j = 0; j <= 8; ++j) {
    const Real uv[] = {0.5, 0.5*Real(j)/8};
    Real pl[3], pr[3], Jl[6], Jr[6];
    s.cell_position1(0, uv, pl, nullptr, nullptr, Jl);
    s.cell_position1(1, uv, pr, nullptr, nullptr, Jr);
    Real err = mv3::distance2(pl, pr);
    for (int k = 0; k < 6; ++k) err = std::max(err, std::abs(Jl[k]-Jr[k]));
    if (err > 1e-12) ++nerr;
  }

  // For testcase 70, refining the grid must reduce the error in both the
  // reconstructed height and its x/y derivatives.
  Real refinement_error[2] = {0, 0};
  for (int ni = 0; ni < 2; ++ni) {
    const int n = ni == 0 ? 4 : 8;
    const auto m = make_grid(n);
    BicubicBezierSurface sr(m, shape);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Idx ci = j*n+i;
        const Real uv[] = {(i + Real(0.37))/n, (j + Real(0.61))/n};
        Real xyz[3], J[6], z, zx, zy;
        sr.cell_position1(ci, uv, xyz, nullptr, nullptr, J);
        gallery::ZxyFn::eval(shape, uv[0], uv[1], z, &zx, &zy);
        refinement_error[ni] = std::max(
          refinement_error[ni],
          std::max(std::abs(xyz[2]-z),
                   std::max(std::abs(J[4]-zx), std::abs(J[5]-zy))));
      }
  }
  if (!(refinement_error[1] < 0.6*refinement_error[0])) {
    printf("BicubicBezierSurface refinement: coarse %1.3e fine %1.3e\n",
           refinement_error[0], refinement_error[1]);
    ++nerr;
  }
  return nerr;
}

} // namespace squirrel
} // namespace woodland
