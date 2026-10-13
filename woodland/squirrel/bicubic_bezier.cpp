#include "woodland/squirrel/bicubic_bezier.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace woodland {
namespace squirrel {

namespace {

inline int idx (const int i, const int j) { return 4*j+i; }

void basis (const Real t, Real b[4], Real db[4]) {
  const Real r = 1-t, r2 = r*r, t2 = t*t;
  b[0] = r*r2;
  b[1] = 3*r2*t;
  b[2] = 3*r*t2;
  b[3] = t*t2;
  db[0] = -3*r2;
  db[1] = 3*r*(r-2*t);
  db[2] = 3*t*(2*r-t);
  db[3] = 3*t2;
}

} // namespace

void BicubicBezier::init_from_hermite (
  const Real hx, const Real hy, CRPtr f, CRPtr fx, CRPtr fy, CRPtr fxy,
  RPtr c)
{
  assert(hx > 0 && hy > 0);
  // Corners.
  c[idx(0,0)] = f[0]; c[idx(3,0)] = f[1];
  c[idx(0,3)] = f[2]; c[idx(3,3)] = f[3];
  // Edge controls from first derivatives.
  c[idx(1,0)] = f[0] + hx*fx[0]/3;
  c[idx(2,0)] = f[1] - hx*fx[1]/3;
  c[idx(1,3)] = f[2] + hx*fx[2]/3;
  c[idx(2,3)] = f[3] - hx*fx[3]/3;
  c[idx(0,1)] = f[0] + hy*fy[0]/3;
  c[idx(0,2)] = f[2] - hy*fy[2]/3;
  c[idx(3,1)] = f[1] + hy*fy[1]/3;
  c[idx(3,2)] = f[3] - hy*fy[3]/3;
  // Interior controls from mixed derivatives.
  const Real hxy9 = hx*hy/9;
  c[idx(1,1)] = c[idx(1,0)] + c[idx(0,1)] - c[idx(0,0)] + hxy9*fxy[0];
  c[idx(2,1)] = c[idx(2,0)] + c[idx(3,1)] - c[idx(3,0)] - hxy9*fxy[1];
  c[idx(1,2)] = c[idx(1,3)] + c[idx(0,2)] - c[idx(0,3)] - hxy9*fxy[2];
  c[idx(2,2)] = c[idx(2,3)] + c[idx(3,2)] - c[idx(3,3)] + hxy9*fxy[3];
}

void BicubicBezier::eval (CRPtr c, const Real u, const Real v, Real& f,
                           RPtr fu, RPtr fv, RPtr fuv) {
  Real bu[4], dbu[4], bv[4], dbv[4];
  basis(u, bu, dbu);
  basis(v, bv, dbv);
  f = 0;
  if (fu) *fu = 0;
  if (fv) *fv = 0;
  if (fuv) *fuv = 0;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) {
      const Real cij = c[idx(i,j)];
      f += bu[i]*bv[j]*cij;
      if (fu) *fu += dbu[i]*bv[j]*cij;
      if (fv) *fv += bu[i]*dbv[j]*cij;
      if (fuv) *fuv += dbu[i]*dbv[j]*cij;
    }
}

int BicubicBezier::unittest () {
  int nerr = 0;
  Real a[16];
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i)
      a[idx(i,j)] = (1 + 2*i - 3*j + i*j)/17.0;
  const auto poly = [&] (const Real x, const Real y, const int dx,
                         const int dy) {
    Real value = 0;
    for (int j = dy; j < 4; ++j)
      for (int i = dx; i < 4; ++i) {
        Real fac = 1;
        for (int k = 0; k < dx; ++k) fac *= i-k;
        for (int k = 0; k < dy; ++k) fac *= j-k;
        value += a[idx(i,j)]*fac*std::pow(x, i-dx)*std::pow(y, j-dy);
      }
    return value;
  };
  const Real x0 = 0.2, y0 = -0.1, hx = 0.7, hy = 0.4;
  Real f[4], fx[4], fy[4], fxy[4];
  const Real xs[4] = {x0, x0+hx, x0, x0+hx};
  const Real ys[4] = {y0, y0, y0+hy, y0+hy};
  for (int k = 0; k < 4; ++k) {
    f[k] = poly(xs[k], ys[k], 0, 0);
    fx[k] = poly(xs[k], ys[k], 1, 0);
    fy[k] = poly(xs[k], ys[k], 0, 1);
    fxy[k] = poly(xs[k], ys[k], 1, 1);
  }
  Real c[16];
  init_from_hermite(hx, hy, f, fx, fy, fxy, c);
  for (int j = 0; j <= 6; ++j)
    for (int i = 0; i <= 6; ++i) {
      const Real u = Real(i)/6, v = Real(j)/6;
      const Real x = x0+hx*u, y = y0+hy*v;
      Real q, qu, qv, quv;
      eval(c, u, v, q, &qu, &qv, &quv);
      const Real err = std::max({
          std::abs(q-poly(x,y,0,0)),
          std::abs(qu/hx-poly(x,y,1,0)),
          std::abs(qv/hy-poly(x,y,0,1)),
          std::abs(quv/(hx*hy)-poly(x,y,1,1))});
      if (err > 1e-11) ++nerr;
    }
  return nerr;
}

} // namespace squirrel
} // namespace woodland
