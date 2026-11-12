#include "woodland/marmot/geometry.hpp"

#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/squirrel/bicubic_bezier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace woodland {
namespace marmot {

typedef acorn::Matvec<3,Real> mv3;
using squirrel::BicubicBezier;

void calc_cubic_spline_slopes (const int n, CRPtr x, CRPtr y, const int ys,
                               RPtr m, const int ms) {
  throw_if(n < 2, "calc_cubic_spline_slopes: need at least 2 knots");
  const auto Y = [&] (const int i) { return y[ys*i]; };
  const auto h = [&] (const int i) { return x[i+1] - x[i]; };
  const auto dd = [&] (const int i) { return (Y(i+1) - Y(i))/h(i); };
  if (n == 2) {
    m[0] = m[ms] = dd(0);
    return;
  }
  if (n == 3) {
    const Real h0 = h(0), h1 = h(1), d0 = dd(0);
    const Real c = (dd(1) - d0)/(h0 + h1);
    m[0] = d0 - c*h0;
    m[ms] = d0 + c*h0;
    m[2*ms] = d0 + c*(h0 + 2*h1);
    return;
  }
  // Tridiagonal system for the slopes; see de Boor, A Practical Guide to
  // Splines, and Matlab's spline.m for the not-a-knot end rows.
  const int N = n-1;
  std::vector<Real> a(n), b(n), c(n), d(n);
  {
    const Real h0 = h(0), h1 = h(1);
    a[0] = 0;
    b[0] = h1;
    c[0] = h0 + h1;
    d[0] = ((h0 + 2*(h0 + h1))*h1*dd(0) + h0*h0*dd(1))/(h0 + h1);
  }
  for (int k = 1; k < N; ++k) {
    a[k] = h(k);
    b[k] = 2*(h(k-1) + h(k));
    c[k] = h(k-1);
    d[k] = 3*(h(k)*dd(k-1) + h(k-1)*dd(k));
  }
  {
    const Real hm2 = h(N-2), hm1 = h(N-1), xn = hm2 + hm1;
    a[N] = xn;
    b[N] = hm2;
    c[N] = 0;
    d[N] = (hm1*hm1*dd(N-2) + (2*xn + hm1)*hm2*dd(N-1))/xn;
  }
  for (int k = 1; k <= N; ++k) {
    const Real w = a[k]/b[k-1];
    b[k] -= w*c[k-1];
    d[k] -= w*d[k-1];
  }
  m[ms*N] = d[N]/b[N];
  for (int k = N-1; k >= 0; --k)
    m[ms*k] = (d[k] - c[k]*m[ms*(k+1)])/b[k];
}

GridGeometry::GridGeometry (const int ni_, const int nj_, CRPtr xyz)
  : ni(ni_), nj(nj_)
{
  throw_if(ni < 2 || nj < 2, "GridGeometry: need ni, nj >= 2");
  init(xyz);
}

void GridGeometry::init (CRPtr xyz) {
  const auto node = [&] (const int i, const int j) { return 3*(j*ni + i); };
  const size_t nn = 3*size_t(ni)*nj;
  P.assign(xyz, xyz + nn);
  Ps.resize(nn); Pt.resize(nn); Pst.resize(nn);
  for (size_t k = 0; k < nn; ++k)
    throw_if(not std::isfinite(P[k]), "GridGeometry: nonfinite point");

  // Chord-length parameters.
  s.resize(ni); t.resize(nj);
  s[0] = t[0] = 0;
  for (int i = 1; i < ni; ++i) {
    Real len = 0;
    for (int j = 0; j < nj; ++j)
      len += std::sqrt(mv3::distance2(&P[node(i-1,j)], &P[node(i,j)]));
    s[i] = s[i-1] + len/nj;
  }
  for (int j = 1; j < nj; ++j) {
    Real len = 0;
    for (int i = 0; i < ni; ++i)
      len += std::sqrt(mv3::distance2(&P[node(i,j-1)], &P[node(i,j)]));
    t[j] = t[j-1] + len/ni;
  }
  for (int i = 1; i < ni; ++i)
    throw_if(not (s[i] - s[i-1] > 1e-12*s[ni-1]),
             "GridGeometry: repeated grid points along i at i = "
             + std::to_string(i));
  for (int j = 1; j < nj; ++j)
    throw_if(not (t[j] - t[j-1] > 1e-12*t[nj-1]),
             "GridGeometry: repeated grid points along j at j = "
             + std::to_string(j));

  // Hermite data of the tensor-product spline: X_s along each i line, X_t
  // along each j line, and X_st as the j-direction spline slopes of X_s.
  for (int j = 0; j < nj; ++j)
    for (int d = 0; d < 3; ++d)
      calc_cubic_spline_slopes(ni, s.data(), &P[node(0,j)+d], 3,
                               &Ps[node(0,j)+d], 3);
  for (int i = 0; i < ni; ++i)
    for (int d = 0; d < 3; ++d) {
      calc_cubic_spline_slopes(nj, t.data(), &P[node(i,0)+d], 3*ni,
                               &Pt[node(i,0)+d], 3*ni);
      calc_cubic_spline_slopes(nj, t.data(), &Ps[node(i,0)+d], 3*ni,
                               &Pst[node(i,0)+d], 3*ni);
    }
}

void GridGeometry::get_domain (Real s_[2], Real t_[2]) const {
  s_[0] = s[0]; s_[1] = s[ni-1];
  t_[0] = t[0]; t_[1] = t[nj-1];
}

static int find_interval (const RealArray& k, const Real x) {
  const int n = int(k.size());
  const auto it = std::upper_bound(k.begin(), k.end(), x);
  const int i = int(it - k.begin()) - 1;
  return std::max(0, std::min(n-2, i));
}

void GridGeometry
::eval (const Real s_, const Real t_, Real X[3], RPtr Xs, RPtr Xt,
        RPtr Xst) const {
  const int i = find_interval(s, s_), j = find_interval(t, t_);
  const Real hs = s[i+1] - s[i], ht = t[j+1] - t[j];
  const Real u = (s_ - s[i])/hs, v = (t_ - t[j])/ht;
  const int ns[4] = {3*(j*ni + i), 3*(j*ni + i+1),
                     3*((j+1)*ni + i), 3*((j+1)*ni + i+1)};
  for (int d = 0; d < 3; ++d) {
    Real f[4], fs[4], ft[4], fst[4], ctrl[16];
    for (int k = 0; k < 4; ++k) {
      f[k] = P[ns[k]+d]; fs[k] = Ps[ns[k]+d];
      ft[k] = Pt[ns[k]+d]; fst[k] = Pst[ns[k]+d];
    }
    BicubicBezier::init_from_hermite(hs, ht, f, fs, ft, fst, ctrl);
    Real fu, fv, fuv;
    BicubicBezier::eval(ctrl, u, v, X[d], &fu, &fv, &fuv);
    if (Xs) Xs[d] = fu/hs;
    if (Xt) Xt[d] = fv/ht;
    if (Xst) Xst[d] = fuv/(hs*ht);
  }
}

GridGeometry::Ptr GridGeometry::read (const std::string& filename) {
  std::ifstream is(filename);
  throw_if(not is.is_open(), "GridGeometry::read: cannot open " + filename);
  std::stringstream ss;
  std::string line;
  while (std::getline(is, line)) {
    const auto pos = line.find('#');
    if (pos != std::string::npos) line.erase(pos);
    ss << line << "\n";
  }
  long ni, nj;
  throw_if(not (ss >> ni >> nj), "GridGeometry::read: missing ni nj in "
           + filename);
  throw_if(ni < 2 || nj < 2 || ni*nj > 100000000L,
           "GridGeometry::read: bad ni nj in " + filename);
  RealArray xyz(3*ni*nj);
  for (auto& e : xyz)
    throw_if(not (ss >> e), "GridGeometry::read: expected 3*ni*nj = "
             + std::to_string(3*ni*nj) + " coordinates in " + filename);
  Real extra;
  throw_if(bool(ss >> extra), "GridGeometry::read: extra data in " + filename);
  return std::make_shared<GridGeometry>(int(ni), int(nj), xyz.data());
}

void GridGeometry::write (const std::string& filename, const int ni,
                          const int nj, CRPtr xyz) {
  FILE* fp = fopen(filename.c_str(), "w");
  throw_if(not fp, "GridGeometry::write: cannot open " + filename);
  fprintf(fp, "# marmot fault geometry: ni nj, then ni*nj points x y z, "
          "i fastest\n%d %d\n", ni, nj);
  for (int k = 0; k < ni*nj; ++k)
    fprintf(fp, "%23.16e %23.16e %23.16e\n",
            xyz[3*k], xyz[3*k+1], xyz[3*k+2]);
  fclose(fp);
}

void ZxyGeometry::get_domain (Real s[2], Real t[2]) const {
  s[0] = 0; s[1] = lx;
  t[0] = 0; t[1] = ly;
}

void ZxyGeometry
::eval (const Real s, const Real t, Real X[3], RPtr Xs, RPtr Xt,
        RPtr Xst) const {
  Real zx, zy, zxy;
  squirrel::gallery::ZxyFn::eval(shape, s/lx, t/ly, X[2], &zx, &zy, &zxy);
  X[0] = s; X[1] = t;
  if (Xs) { Xs[0] = 1; Xs[1] = 0; Xs[2] = zx/lx; }
  if (Xt) { Xt[0] = 0; Xt[1] = 1; Xt[2] = zy/ly; }
  if (Xst) { Xst[0] = 0; Xst[1] = 0; Xst[2] = zxy/(lx*ly); }
}

static int test_spline_slopes () {
  int nerr = 0;
  for (int n = 2; n <= 7; ++n) {
    // Nonuniform knots; the spline must reproduce polynomials of degree
    // min(3, n-1).
    Real x[7], y[7], m[7];
    for (int i = 0; i < n; ++i) x[i] = i + 0.3*std::sin(1.7*i);
    const int deg = std::min(3, n-1);
    const Real c[] = {0.4, -1.3, 0.7, 0.25};
    for (int i = 0; i < n; ++i) {
      y[i] = 0;
      for (int p = 0; p <= deg; ++p) y[i] += c[p]*std::pow(x[i], p);
    }
    calc_cubic_spline_slopes(n, x, y, 1, m, 1);
    for (int i = 0; i < n; ++i) {
      Real yp = 0;
      for (int p = 1; p <= deg; ++p) yp += p*c[p]*std::pow(x[i], p-1);
      if (std::abs(m[i] - yp) > 1e-12*(1 + std::abs(yp))) {
        printf("spline slopes: n %d i %d m %1.6e exact %1.6e\n",
               n, i, m[i], yp);
        ++nerr;
      }
    }
  }
  return nerr;
}

static int test_planar_grid () {
  // Nonuniform grid on a tilted plane: chord-length parameters are exact
  // distances, so the spline reproduces the plane, the parameterization, and
  // its derivatives exactly.
  int nerr = 0;
  const int ni = 6, nj = 5;
  Real e1[] = {0.3, -0.5, 0.8}, e2[3], o[] = {1, -2, -3};
  mv3::normalize(e1);
  {
    const Real tmp[] = {0.1, 0.9, 0.2};
    Real n[3];
    mv3::cross(e1, tmp, n);
    mv3::cross(n, e1, e2);
    mv3::normalize(e2);
  }
  Real si[ni], tj[nj];
  for (int i = 0; i < ni; ++i) si[i] = 2*i + 0.4*std::sin(1.1*i);
  for (int j = 0; j < nj; ++j) tj[j] = 1.5*j + 0.3*std::cos(0.7*j) - 0.3;
  RealArray xyz(3*ni*nj);
  for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i)
      for (int d = 0; d < 3; ++d)
        xyz[3*(j*ni+i)+d] = o[d] + si[i]*e1[d] + tj[j]*e2[d];
  GridGeometry g(ni, nj, xyz.data());
  for (int k = 0; k < 50; ++k) {
    const Real s = (si[ni-1] - si[0])*(k % 7)/6.0;
    const Real t = (tj[nj-1] - tj[0])*(k % 9)/8.0;
    Real X[3], Xs[3], Xt[3], Xst[3];
    g.eval(s, t, X, Xs, Xt, Xst);
    Real err = 0;
    for (int d = 0; d < 3; ++d) {
      const Real Xe = o[d] + (s + si[0])*e1[d] + (t + tj[0])*e2[d];
      err = std::max({err, std::abs(X[d] - Xe), std::abs(Xs[d] - e1[d]),
                      std::abs(Xt[d] - e2[d]), std::abs(Xst[d])});
    }
    if (err > 1e-12) {
      printf("GridGeometry planar: s %1.3e t %1.3e err %1.3e\n", s, t, err);
      ++nerr;
    }
  }
  // Interpolation of the input points.
  for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i) {
      Real X[3];
      g.eval(g.get_s()[i], g.get_t()[j], X);
      if (std::sqrt(mv3::distance2(X, &xyz[3*(j*ni+i)])) > 1e-12)
        ++nerr;
    }
  return nerr;
}

static int test_convergence () {
  // Sample an analytic curved surface on increasingly fine grids; the error of
  // the spline surface must decrease at roughly fourth order.
  using squirrel::gallery::ZxyFn;
  int nerr = 0;
  const auto shape = ZxyFn::Shape::cos10_cos12_y;
  const Real amp0 = ZxyFn::get_cos10_cos12_amp();
  ZxyFn::set_cos10_cos12_amp(0.05);
  Real errs[2];
  for (int r = 0; r < 2; ++r) {
    const int n = r == 0 ? 12 : 24;
    RealArray xyz(3*(n+1)*(n+1));
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i) {
        auto p = &xyz[3*(j*(n+1)+i)];
        p[0] = Real(i)/n; p[1] = Real(j)/n;
        ZxyFn::eval(shape, p[0], p[1], p[2], nullptr, nullptr);
      }
    GridGeometry g(n+1, n+1, xyz.data());
    Real s[2], t[2];
    g.get_domain(s, t);
    errs[r] = 0;
    for (int j = 0; j < 40; ++j)
      for (int i = 0; i < 40; ++i) {
        Real X[3], z;
        g.eval(s[0] + (s[1]-s[0])*(i + 0.5)/40, t[0] + (t[1]-t[0])*(j + 0.5)/40,
               X);
        ZxyFn::eval(shape, X[0], X[1], z, nullptr, nullptr);
        errs[r] = std::max(errs[r], std::abs(X[2] - z));
      }
  }
  ZxyFn::set_cos10_cos12_amp(amp0);
  const Real ooa = std::log2(errs[0]/errs[1]);
  if (not (ooa > 3.5)) {
    printf("GridGeometry convergence: err %1.3e %1.3e ooa %1.2f\n",
           errs[0], errs[1], ooa);
    ++nerr;
  }
  return nerr;
}

int GridGeometry::unittest () {
  return test_spline_slopes() + test_planar_grid() + test_convergence();
}

} // namespace marmot
} // namespace woodland
