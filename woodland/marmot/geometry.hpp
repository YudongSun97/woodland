#ifndef INCLUDE_WOODLAND_MARMOT_GEOMETRY
#define INCLUDE_WOODLAND_MARMOT_GEOMETRY

#include "woodland/marmot/marmot.hpp"
#include "woodland/squirrel/gallery.hpp"

#include <memory>
#include <string>

namespace woodland {
namespace marmot {

// A smooth parametric fault surface X(s,t) in 3D, (s,t) in [s0,s1] x [t0,t1].
// The surface need not be a graph over any coordinate plane: it can be
// vertical, overturned, or curve in every direction.
struct Geometry {
  typedef std::shared_ptr<Geometry> Ptr;
  typedef std::shared_ptr<const Geometry> CPtr;

  virtual ~Geometry () {}

  virtual const char* name () const { return "Geometry"; }

  // Parameter domain.
  virtual void get_domain(Real s[2], Real t[2]) const = 0;

  // Position X and its derivatives X_s, X_t, X_st, each a 3-vector. Derivative
  // outputs may be nullptr. (s,t) may lie somewhat outside the domain, in which
  // case the representation is extrapolated.
  virtual void eval(const Real s, const Real t, Real X[3],
                    RPtr Xs = nullptr, RPtr Xt = nullptr,
                    RPtr Xst = nullptr) const = 0;
};

// Fault geometry given as a structured (ni x nj) grid of 3D points P(i,j), i
// fastest, ni, nj >= 2. The grid lines need not be planar or straight. The grid
// is interpolated by a C2 tensor-product cubic spline (not-a-knot end
// conditions) in chord-length parameters:
//   s_i = mean over j of the polyline length along i from P(0,j) to P(i,j),
//   t_j = mean over i of the polyline length along j from P(i,0) to P(i,j).
// Thus (s,t) are approximately arc length, in the units of the input.
struct GridGeometry : public Geometry {
  typedef std::shared_ptr<GridGeometry> Ptr;

  GridGeometry(const int ni, const int nj, CRPtr xyz);

  // Read from a text file. '#' starts a comment. The first two numbers are ni
  // and nj, followed by ni*nj triples x y z, i fastest.
  static Ptr read(const std::string& filename);
  // Write in the format read() reads.
  static void write(const std::string& filename, const int ni, const int nj,
                    CRPtr xyz);

  const char* name () const override { return "GridGeometry"; }

  void get_domain(Real s[2], Real t[2]) const override;
  void eval(const Real s, const Real t, Real X[3], RPtr Xs = nullptr,
            RPtr Xt = nullptr, RPtr Xst = nullptr) const override;

  int get_ni () const { return ni; }
  int get_nj () const { return nj; }
  // Spline knots, which are the parameters of the input points.
  const RealArray& get_s () const { return s; }
  const RealArray& get_t () const { return t; }

  static int unittest();

private:
  int ni, nj;
  RealArray s, t;
  // Nodal Hermite data, 3 reals per node, node (i,j) at 3*(j*ni + i).
  RealArray P, Ps, Pt, Pst;

  void init(CRPtr xyz);
};

// First derivatives m[0:n] at the knots x[0:n] of the C2 cubic spline
// interpolating y[0:n] with not-a-knot end conditions. y and m are accessed
// with stride. n = 2 gives the line, n = 3 the parabola.
void calc_cubic_spline_slopes(const int n, CRPtr x, CRPtr y, const int ystride,
                              RPtr m, const int mstride);

// Analytic surface z = f(x/lx, y/ly) from squirrel::gallery::ZxyFn, with
// (s,t) = (x,y) in [0,lx] x [0,ly]. Used to compare with squirrel.
struct ZxyGeometry : public Geometry {
  ZxyGeometry (const squirrel::gallery::ZxyFn::Shape shape_,
               const Real lx_ = 1, const Real ly_ = 1)
    : shape(shape_), lx(lx_), ly(ly_)
  {}

  const char* name () const override { return "ZxyGeometry"; }

  void get_domain(Real s[2], Real t[2]) const override;
  void eval(const Real s, const Real t, Real X[3], RPtr Xs = nullptr,
            RPtr Xt = nullptr, RPtr Xst = nullptr) const override;

private:
  squirrel::gallery::ZxyFn::Shape shape;
  Real lx, ly;
};

} // namespace marmot
} // namespace woodland

#endif
