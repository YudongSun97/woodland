#include "woodland/squirrel/ctzx.hpp"

#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/dbg.hpp"

#include <cmath>

namespace woodland {
namespace squirrel {
namespace ctzx {

typedef acorn::Matvec<3,Real> mv3;

struct ExactData : public Exact::Description {
  ExactData (const ZxFn::Shape zshape_, const Disloc::CPtr& disloc_)
    : zshape(zshape_), disloc(disloc_)
  {}

  void get_rectangle_limits (Real& xlo, Real& xhi, Real& ylo, Real& yhi)
    const override
  {
    xlo = ylo = 0;
    xhi = yhi = 1;
  }
  
  void get_surface (const Real x, const Real y,
                    Real& z, Real z_xy[2], Real lcs[9]) const override {
    Real g;
    eval(zshape, x, z, g);
    if (z_xy) {
      z_xy[0] = g;
      z_xy[1] = 0;
    }
    if (lcs) {
      for (int i = 0; i < 9; ++i) lcs[i] = 0;
      lcs[4] = 1;
      lcs[6] = -g;
      lcs[8] = 1;
      mv3::normalize(&lcs[6]);
      mv3::cross(&lcs[3], &lcs[6], &lcs[0]);
    }
  }

  void get_disloc (const Real x, const Real y, Real d[3]) const override {
    const Real p[] = {x, y};
    disloc->eval(p, d);
  }

private:
  ZxFn::Shape zshape;
  Disloc::CPtr disloc;
};

// Yudong 03/04/2026
// Exact description for z = f(x,y) surfaces. LCS and get_disloc match ExactData.
struct ExactDataZxy : public Exact::Description {
  ExactDataZxy (const gallery::ZxyFn::Shape zxy_shape_, const Disloc::CPtr& disloc_)
    : zxy_shape(zxy_shape_), disloc(disloc_)
  {}

  void get_rectangle_limits (Real& xlo, Real& xhi, Real& ylo, Real& yhi)
    const override
  {
    xlo = ylo = 0;
    xhi = yhi = 1;
  }

  void get_surface (const Real x, const Real y,
                    Real& z, Real z_xy[2], Real lcs[9]) const override {
    Real zx = 0, zy = 0;
    gallery::ZxyFn::eval(zxy_shape, x, y, z,
                         z_xy ? &zx : nullptr,
                         z_xy ? &zy : nullptr);
    const Real zy_tol = 1e-14;
    const bool z_f_of_x = (std::fabs(zy) <= zy_tol);
    if (z_f_of_x && zxy_shape == gallery::ZxyFn::Shape::trig1_cosy) {
      // Use same 1D eval as ExactData(trig1) so z,zx are bit-identical.
      gallery::eval(ZxFn::Shape::trig1, x, z, zx);
      zy = 0;
    }
    if (z_xy) {
      z_xy[0] = zx;
      z_xy[1] = zy;
    }
    if (lcs){
      Real n[3] = {-zx, -zy, 1};
      mv3::normalize(n);
      Real t1[3] = {0, 1, zy};
      mv3::normalize(t1);
      Real t2[3];
      mv3::cross(t1, n, t2);
      mv3::normalize(t2);
      lcs[0] = t2[0]; lcs[1] = t2[1]; lcs[2] = t2[2];
      lcs[3] = t1[0]; lcs[4] = t1[1]; lcs[5] = t1[2];
      lcs[6] = n[0];  lcs[7] = n[1];  lcs[8] = n[2];
    }
  }

  void get_disloc (const Real x, const Real y, Real d[3]) const override {
    const Real p[] = {x, y};
    disloc->eval(p, d);
  }

private:
  gallery::ZxyFn::Shape zxy_shape;
  Disloc::CPtr disloc;
};

static void setup (ConvTest& ct, Exact::Ptr& exact, Exact::Options* co) {
  if (not ct.get_discretization()) ct.discretize();

  // Yudong 03/04/2026
  if (not exact) exact = std::make_shared<Exact>();
  Exact::Description::CPtr description;
  if (ct.get_use_zxy()) {
    description = std::make_shared<ExactDataZxy>(
      ct.get_zxy_shape(), ct.get_disloc());
  } else {
    description = std::make_shared<ExactData>(
      ct.get_zxfn()->get_shape(), ct.get_disloc());
  }
  exact->init(description);
  exact->set_lam_mu(ct.get_lam(), ct.get_mu());
  exact->set_halfspace(ct.get_use_halfspace());

  if (co)
    exact->set_options(*co);
  else {
    Exact::Options o;
    o.nxr = 11; o.nyr = 11;
    o.np_radial = 20; o.np_angular = 20;
    o.triquad_order = -1;
    exact->set_options(o);
  }
}

void ConvTest::eval_exact_at_cell_ctrs (RealArray& sigmas, Exact::Options* o) {
  setup(*this, exact, o);
  const auto& m = *d->get_mesh();
  const auto& srf = *d->get_surface();
  const auto ncell = m.get_ncell();
  sigmas.resize(6*ncell);
  ompparfor for (Idx ir = 0; ir < ncell; ++ir) {
    if (verbosity > 0 && ir / (ncell/10) != (ir-1) / (ncell/10)) {
      printf(" %3.1f", 100*Real(ir)/ncell);
      fflush(stdout);
    }
    auto* const sigma = &sigmas[6*ir];
    Real xy[3];
    srf.cell_ctr_xyz(ir, xy);
    exact->calc_stress(xy[0], xy[1], sigma);
  }
  if (verbosity > 0) printf("\n");
}

void ConvTest
::eval_exact_at_rect_ctrs (const int nxr, const int nyr, RealArray& sigmas,
                           Exact::Options* o) {
  setup(*this, exact, o);
  ZxFn zxfno(zxfn->get_shape());
  zxfno.set_nx(nxr);
  const int ncell = nxr*nyr;
  sigmas.resize(6*ncell);
  ompparfor for (Idx ir = 0; ir < ncell; ++ir) {
    if (verbosity > 0 && ir / (ncell/10) != (ir-1) / (ncell/10)) {
      printf(" %3.1f", 100*Real(ir)/ncell);
      fflush(stdout);
    }
    auto* const sigma = &sigmas[6*ir];
    Real xy[2];
    calc_rect_ctr(zxfno, nyr, ir, xy);
    exact->calc_stress(xy[0], xy[1], sigma);
  }
  if (verbosity > 0) printf("\n");
}

} // namespace ctzx
} // namespace squirrel
} // namespace woodland
