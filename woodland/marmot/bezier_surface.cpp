#include "woodland/marmot/bezier_surface.hpp"

#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/squirrel/bicubic_bezier.hpp"
#include "woodland/squirrel/mesh.hpp"

#include <algorithm>
#include <cmath>

namespace woodland {
namespace marmot {

typedef acorn::Matvec<3,Real> mv3;
using squirrel::BicubicBezier;
namespace mesh = squirrel::mesh;

RealArray uniform_breaks (const Real a, const Real b, const int n) {
  throw_if(n < 1, "uniform_breaks: n must be positive");
  RealArray x(n+1);
  for (int i = 0; i <= n; ++i) x[i] = a + (b - a)*Real(i)/n;
  x[n] = b;
  return x;
}

BezierPatchSurface::LcsPrimary
BezierPatchSurface::convert (const std::string& s) {
  if (s == "strike") return LcsPrimary::strike;
  if (s == "x") return LcsPrimary::x;
  if (s == "s") return LcsPrimary::s;
  throw_if(true, "BezierPatchSurface: lcs must be strike, x, or s; got " + s);
  return LcsPrimary::strike;
}

std::string BezierPatchSurface::convert (const LcsPrimary p) {
  switch (p) {
  case LcsPrimary::x: return "x";
  case LcsPrimary::s: return "s";
  case LcsPrimary::strike:
  default: return "strike";
  }
}

BezierPatchSurface
::BezierPatchSurface (const Geometry& g, const RealArray& sb_,
                      const RealArray& tb_, const LcsPrimary primary_)
  : nx(int(sb_.size()) - 1), ny(int(tb_.size()) - 1), sb(sb_), tb(tb_),
    primary(primary_)
{
  throw_if(nx < 1 || ny < 1, "BezierPatchSurface: need >= 2 breakpoints");
  for (int i = 0; i < nx; ++i)
    throw_if(not (sb[i+1] > sb[i]), "BezierPatchSurface: sb must increase");
  for (int j = 0; j < ny; ++j)
    throw_if(not (tb[j+1] > tb[j]), "BezierPatchSurface: tb must increase");
  init(g);
}

// Tangent-plane projection of p, normalized. Returns false if p is (nearly)
// normal to the plane.
static bool project_to_plane (const Real zhat[3], const Real p[3],
                              Real xhat[3]) {
  const Real pn = mv3::norm2(p);
  if (pn == 0) return false;
  mv3::axpbyz(1, p, -mv3::dot(zhat, p), zhat, xhat);
  const Real xn = mv3::norm2(xhat);
  if (xn <= 1e-6*pn) return false;
  mv3::scale(1/xn, xhat);
  return true;
}

void BezierPatchSurface::init (const Geometry& g) {
  const int nsn = nx+1, ntn = ny+1;
  const Idx nc = get_ncell();
  const auto node = [&] (const int i, const int j) { return 3*(j*nsn + i); };

  // Nodal Hermite data from the geometry.
  RealArray X(3*nsn*ntn), Xs(X.size()), Xt(X.size()), Xst(X.size());
  ompparfor for (int k = 0; k < nsn*ntn; ++k) {
    const int i = k % nsn, j = k / nsn;
    g.eval(sb[i], tb[j], &X[3*k], &Xs[3*k], &Xt[3*k], &Xst[3*k]);
  }
  for (const auto e : X)
    throw_if(not std::isfinite(e), "BezierPatchSurface: nonfinite geometry");

  // The curved rectangular mesh.
  const auto vtxs = mesh::make_vertices(3);
  for (int k = 0; k < nsn*ntn; ++k) vtxs->append_vtx(&X[3*k]);
  const auto topo = mesh::make_topology();
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const Idx cell[] = {Idx(j*nsn + i), Idx(j*nsn + i+1),
                          Idx((j+1)*nsn + i+1), Idx((j+1)*nsn + i)};
      topo->append_cell(4, cell);
    }
  topo->end_modifications();
  m = mesh::make_mesh(vtxs, topo, mesh::make_relations(topo));

  // Patches. Corner order for init_from_hermite is (00, 10, 01, 11).
  ctrl.resize(48*nc);
  xyz_ctrs.resize(3*nc);
  lcs_ctrs.resize(9*nc);
  primaries.resize(3*nc);
  ompparfor for (Idx ci = 0; ci < nc; ++ci) {
    int i, j;
    cell_ij(ci, i, j);
    const Real hs = sb[i+1] - sb[i], ht = tb[j+1] - tb[j];
    const int ns[4] = {node(i,j), node(i+1,j), node(i,j+1), node(i+1,j+1)};
    for (int d = 0; d < 3; ++d) {
      Real f[4], fs[4], ft[4], fst[4];
      for (int k = 0; k < 4; ++k) {
        f[k] = X[ns[k]+d]; fs[k] = Xs[ns[k]+d];
        ft[k] = Xt[ns[k]+d]; fst[k] = Xst[ns[k]+d];
      }
      BicubicBezier::init_from_hermite(hs, ht, f, fs, ft, fst,
                                       &ctrl[48*ci + 16*d]);
    }

    // Center LCS. First get the normal and X_s; then fix the primary
    // direction, which cell_position uses at every point in the cell.
    Real uv[2], J[6], zhat[3];
    cell_ctr_uv(ci, uv);
    cell_position(ci, 1, uv, &xyz_ctrs[3*ci], nullptr, nullptr, J);
    const Real a[] = {J[0], J[2], J[4]}, b[] = {J[1], J[3], J[5]};
    mv3::cross(a, b, zhat);
    mv3::normalize(zhat);
    Real p[3], xhat[3];
    switch (primary) {
    case LcsPrimary::strike: {
      p[0] = -zhat[1]; p[1] = zhat[0]; p[2] = 0;
    } break;
    case LcsPrimary::x: {
      p[0] = 1; p[1] = 0; p[2] = 0;
    } break;
    case LcsPrimary::s:
    default:
      mv3::copy(a, p);
    }
    if (not project_to_plane(zhat, p, xhat)) {
      const bool ok = project_to_plane(zhat, a, xhat);
      throw_if(not ok, "BezierPatchSurface: degenerate cell " +
               std::to_string(ci));
    }
    mv3::copy(xhat, &primaries[3*ci]);
    cell_position(ci, 1, uv, &xyz_ctrs[3*ci], &lcs_ctrs[9*ci]);
  }
}

void BezierPatchSurface::cell_vtxs_uv (const Idx ci, RPtr uv) const {
  int i, j;
  cell_ij(ci, i, j);
  uv[0] = sb[i  ]; uv[1] = tb[j  ];
  uv[2] = sb[i+1]; uv[3] = tb[j  ];
  uv[4] = sb[i+1]; uv[5] = tb[j+1];
  uv[6] = sb[i  ]; uv[7] = tb[j+1];
}

void BezierPatchSurface::cell_ctr_uv (const Idx ci, Real uv[2]) const {
  int i, j;
  cell_ij(ci, i, j);
  uv[0] = 0.5*(sb[i] + sb[i+1]);
  uv[1] = 0.5*(tb[j] + tb[j+1]);
}

void BezierPatchSurface::cell_ctr_xyz (const Idx ci, Real xyz[3]) const {
  mv3::copy(&xyz_ctrs[3*ci], xyz);
}

void BezierPatchSurface::cell_ctr_lcs (const Idx ci, Real lcs[9]) const {
  acorn::copy(9, &lcs_ctrs[9*ci], lcs);
}

void BezierPatchSurface
::cell_position (const Idx ci, const int n, CRPtr uv, RPtr xyz_, RPtr lcs_,
                 RPtr jacdet, RPtr jac) const {
  assert(ci >= 0 && ci < get_ncell());
  int i, j;
  cell_ij(ci, i, j);
  const Real s0 = sb[i], hs = sb[i+1] - s0, t0 = tb[j], ht = tb[j+1] - t0;
  const Real* const c = &ctrl[48*ci];
  for (int k = 0; k < n; ++k) {
    const Real u = (uv[2*k] - s0)/hs, v = (uv[2*k+1] - t0)/ht;
    Real Xs[3], Xt[3];
    auto xyz = &xyz_[3*k];
    for (int d = 0; d < 3; ++d) {
      Real fu, fv;
      BicubicBezier::eval(c + 16*d, u, v, xyz[d], &fu, &fv);
      Xs[d] = fu/hs;
      Xt[d] = fv/ht;
    }
    if (jac) {
      auto J = &jac[6*k];
      for (int d = 0; d < 3; ++d) {
        J[2*d] = Xs[d];
        J[2*d+1] = Xt[d];
      }
    }
    if (not (jacdet or lcs_)) continue;
    Real nml[3];
    mv3::cross(Xs, Xt, nml);
    const Real area = mv3::norm2(nml);
    if (jacdet) jacdet[k] = area;
    if (lcs_) {
      auto lcs = &lcs_[9*k];
      const auto zhat = lcs + 6;
      mv3::copy(nml, zhat);
      mv3::scale(1/area, zhat);
      if (not project_to_plane(zhat, &primaries[3*ci], lcs))
        project_to_plane(zhat, Xs, lcs);
      mv3::cross(zhat, lcs, lcs + 3);
      mv3::normalize(lcs + 3);
    }
  }
}

bool BezierPatchSurface
::calc_cell_lcs (const Idx ci, const Real p[3], Real uv[2]) const {
  cell_ctr_uv(ci, uv);
  if (mv3::distance2(p, &xyz_ctrs[3*ci]) == 0) return true;
  int i, j;
  cell_ij(ci, i, j);
  const Real h = std::max(sb[i+1] - sb[i], tb[j+1] - tb[j]);
  // Gauss-Newton for min |X(uv) - p|^2.
  for (int it = 0; it < 30; ++it) {
    Real X[3], J[6], r[3];
    cell_position(ci, 1, uv, X, nullptr, nullptr, J);
    mv3::subtract(p, X, r);
    Real A[3] = {0}, b[2] = {0};
    for (int d = 0; d < 3; ++d) {
      A[0] += J[2*d]*J[2*d];
      A[1] += J[2*d]*J[2*d+1];
      A[2] += J[2*d+1]*J[2*d+1];
      b[0] += J[2*d]*r[d];
      b[1] += J[2*d+1]*r[d];
    }
    const Real det = A[0]*A[2] - A[1]*A[1];
    if (not (det > 0)) return false;
    Real du = ( A[2]*b[0] - A[1]*b[1])/det;
    Real dv = (-A[1]*b[0] + A[0]*b[1])/det;
    // Keep steps bounded so the polynomial extension is not followed far.
    const Real step = std::sqrt(du*du + dv*dv), smax = h;
    if (step > smax) { du *= smax/step; dv *= smax/step; }
    uv[0] += du;
    uv[1] += dv;
    if (step <= 1e-14*h) return true;
  }
  return false;
}

static int test_continuity_and_area () {
  // A quarter cylinder of radius R and height H, given as a grid whose lines
  // are circles and verticals: not a graph z = f(x,y). Mesh breakpoints do not
  // coincide with the geometry grid.
  int nerr = 0;
  const Real R = 1.3, H = 0.8;
  const int ni = 25, nj = 9;
  RealArray xyz(3*ni*nj);
  for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i) {
      const Real th = 0.5*M_PI*i/(ni-1);
      auto p = &xyz[3*(j*ni+i)];
      p[0] = R*std::cos(th); p[1] = -H*j/(nj-1); p[2] = R*std::sin(th) - 2;
    }
  GridGeometry g(ni, nj, xyz.data());
  Real sd[2], td[2];
  g.get_domain(sd, td);
  const int nx = 7, ny = 3;
  BezierPatchSurface srf(g, uniform_breaks(sd[0], sd[1], nx),
                         uniform_breaks(td[0], td[1], ny));
  // Position and tangents agree on shared edges.
  Real maxerr = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const Idx ci = j*nx + i;
      for (int e = 0; e < 2; ++e) {
        if ((e == 0 && i+1 == nx) || (e == 1 && j+1 == ny)) continue;
        const Idx cn = e == 0 ? ci + 1 : ci + nx;
        for (int k = 0; k <= 6; ++k) {
          const Real a = Real(k)/6;
          const Real uv[] = {
            e == 0 ? srf.get_sb()[i+1] :
                     srf.get_sb()[i] + a*(srf.get_sb()[i+1] - srf.get_sb()[i]),
            e == 1 ? srf.get_tb()[j+1] :
                     srf.get_tb()[j] + a*(srf.get_tb()[j+1] - srf.get_tb()[j])};
          Real p0[3], p1[3], J0[6], J1[6];
          srf.cell_position1(ci, uv, p0, nullptr, nullptr, J0);
          srf.cell_position1(cn, uv, p1, nullptr, nullptr, J1);
          maxerr = std::max(maxerr, std::sqrt(mv3::distance2(p0, p1)));
          for (int q = 0; q < 6; ++q)
            maxerr = std::max(maxerr, std::abs(J0[q] - J1[q]));
        }
      }
    }
  if (maxerr > 1e-12) {
    printf("BezierPatchSurface continuity: err %1.3e\n", maxerr);
    ++nerr;
  }
  // Points lie on the cylinder, LCS is orthonormal with zhat along the
  // normal, and the area is accurate, to the O(h^4) accuracy of the cubic
  // patches (about 1e-5 here).
  acorn::Workspace w;
  Real area = 0, rerr = 0, oerr = 0, nml_err = 0;
  for (Idx ci = 0; ci < srf.get_ncell(); ++ci) {
    area += squirrel::cell_surface_area(w, srf, ci);
    Real uv[2], p[3], lcs[9];
    srf.cell_ctr_uv(ci, uv);
    uv[0] += 0.1*(srf.get_sb()[1] - srf.get_sb()[0]);
    srf.cell_position1(ci, uv, p, lcs);
    const Real r = std::sqrt(p[0]*p[0] + (p[2]+2)*(p[2]+2));
    rerr = std::max(rerr, std::abs(r - R));
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        oerr = std::max(oerr, std::abs(mv3::dot(lcs+3*a, lcs+3*b) -
                                       (a == b ? 1 : 0)));
    // Radial normal (up to sign).
    const Real nrad[] = {p[0]/r, 0, (p[2]+2)/r};
    nml_err = std::max(nml_err, 1 - std::abs(mv3::dot(nrad, lcs+6)));
    // Strike xhat is horizontal.
    oerr = std::max(oerr, std::abs(lcs[2]));
  }
  const Real area_true = 0.5*M_PI*R*H;
  if (std::abs(area - area_true) > 1e-4*area_true || rerr > 1e-4 ||
      oerr > 1e-13 || nml_err > 1e-8) {
    printf("BezierPatchSurface cylinder: area %1.10e true %1.10e rerr %1.3e"
           " oerr %1.3e nerr %1.3e\n", area, area_true, rerr, oerr, nml_err);
    ++nerr;
  }
  // Closest point inverse.
  for (Idx ci = 0; ci < srf.get_ncell(); ++ci) {
    Real uv0[2], uv[2], p[3];
    srf.cell_ctr_uv(ci, uv0);
    uv0[0] += 0.3*(srf.get_sb()[1] - srf.get_sb()[0]);
    uv0[1] -= 0.2*(srf.get_tb()[1] - srf.get_tb()[0]);
    srf.cell_position1(ci, uv0, p);
    if (not srf.calc_cell_lcs(ci, p, uv) ||
        std::abs(uv[0] - uv0[0]) + std::abs(uv[1] - uv0[1]) > 1e-10) {
      printf("BezierPatchSurface calc_cell_lcs failed for ci %d\n", int(ci));
      ++nerr;
    }
  }
  return nerr;
}

int BezierPatchSurface::unittest () {
  return test_continuity_and_area();
}

} // namespace marmot
} // namespace woodland
