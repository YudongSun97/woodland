#ifndef INCLUDE_WOODLAND_SQUIRREL_BICUBIC_BEZIER
#define INCLUDE_WOODLAND_SQUIRREL_BICUBIC_BEZIER

#include "woodland/squirrel/squirrel.hpp"

namespace woodland {
namespace squirrel {

// Standard tensor-product cubic Bezier scalar patch. Control values are stored
// with the u/x index fastest: ctrl[4*j+i] is Z_ij.
struct BicubicBezier {
  // Corner data are ordered (00, 10, 01, 11). fx, fy, and fxy are derivatives
  // with respect to physical coordinates whose cell widths are hx and hy.
  static void init_from_hermite(
    const Real hx, const Real hy, CRPtr f, CRPtr fx, CRPtr fy, CRPtr fxy,
    RPtr ctrl);

  // Evaluate at local patch coordinates (u,v) in [0,1]^2. Derivatives are
  // with respect to u and v.
  static void eval(CRPtr ctrl, const Real u, const Real v, Real& f,
                   RPtr fu = nullptr, RPtr fv = nullptr,
                   RPtr fuv = nullptr);

  static int unittest();
};

} // namespace squirrel
} // namespace woodland

#endif
