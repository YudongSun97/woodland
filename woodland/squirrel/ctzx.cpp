#include "woodland/acorn/compose_quadrature.hpp"
#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/triangle.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/acorn/fs3d.hpp"
#include "woodland/acorn/dbg.hpp"

#include "woodland/squirrel/mesh.hpp"
#include "woodland/squirrel/global_y_surface.hpp"
#include "woodland/squirrel/global_z_surface.hpp"
#include "woodland/squirrel/bicubic_bezier_surface.hpp"
#include "woodland/squirrel/ctzx.hpp"
#include "woodland/squirrel/pywrite.hpp"
#include "woodland/squirrel/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <map>
#include <stdexcept>

namespace woodland {
namespace squirrel {
namespace ctzx {

typedef acorn::Matvec<2,Real> mv2;
typedef acorn::Matvec<3,Real> mv3;

namespace {

static const bool use_calc_integral_tensor_quadrature = true;

void pywrite (FILE* fp, const std::string& dict, const Triangulation& t) {
  fprintf(fp, "%s['topo'] = npy.array([\n", dict.c_str());
  const auto ntri = t.get_ncell();
  for (int ti = 0; ti < ntri; ++ti) {
    const auto tri = t.get_tri(ti);
    fprintf(fp, "[%d, %d, %d],\n", int(tri[0]), int(tri[1]), int(tri[2]));
  }
  fprintf(fp, "], dtype=npy.int32)\n");
  fprintf(fp, "%s['vtx'] = npy.array([\n", dict.c_str());
  const auto nvtx = t.get_nvtx();
  for (int vi = 0; vi < nvtx; ++vi) {
    const auto vtx = t.get_vtx(vi);
    fprintf(fp, "[%15.8e, %15.8e, %15.8e],\n", vtx[0], vtx[1], vtx[2]);
  }
  fprintf(fp, "])\n");
}

void pywrite (FILE* fp, const std::string& dict, const mesh::Mesh& m) {
  fprintf(fp, "%s['topo'] = npy.array([\n", dict.c_str());
  const auto nc = m.get_ncell();
  const auto to = *m.get_topo();
  const auto v = *m.get_vtxs();
  int nv = 0;
  for (int ci = 0; ci < nc; ++ci) nv = std::max(nv, to.get_cell_nvtx(ci));
  for (int ci = 0; ci < nc; ++ci) {
    const auto c = to.get_cell(ci);
    fprintf(fp, "[");
    for (int i = 0; i < c.n; ++i) fprintf(fp, "%d,", int(c[i]));
    for (int i = c.n; i < nv; ++i) fprintf(fp, "%d,", int(c[c.n-1]));
    fprintf(fp, "],\n");
  }
  fprintf(fp, "], dtype=npy.int32)\n");
  fprintf(fp, "%s['vtx'] = npy.array([\n", dict.c_str());
  const auto nvtx = v.get_nvtx();
  for (int vi = 0; vi < nvtx; ++vi) {
    const auto vtx = v.get_vtx(vi);
    fprintf(fp, "[%15.8e, %15.8e, %15.8e],\n", vtx[0], vtx[1], vtx[2]);
  }
  fprintf(fp, "])\n");
}

} // namespace

void ConvTest::init (const ZxFn::Shape shape, const Disloc::CPtr& disloc_) {
  use_zxy = false;
  use_yxz = false;
  zxfn = std::make_shared<ZxFn>(shape);
  disloc = disloc_;
  gfp.lam = gfp.mu = 1;
  gfp.halfspace = false;
#ifndef WOODLAND_ACORN_HAVE_DC3D
  use_woodland_rg0c0 = true;
#endif
}
// Yudong 02/23/2026
void ConvTest::init_zxy (const gallery::ZxyFn::Shape shape,
                         const Disloc::CPtr& disloc_,
                         const Real lx, const Real ly) {
  throw_if(lx <= 0 || ly <= 0, "init_zxy: lx and ly must be positive");
  use_zxy = true;
  use_yxz = false;
  zxy_shape = shape;
  zxy_lx = lx;
  zxy_ly = ly;
  dargs.zxy_lx = lx;
  dargs.zxy_ly = ly;
  zxfn.reset();
  disloc = disloc_;
  nx_zxy = 2;
  ny_zxy = 2;
  gfp.lam = gfp.mu = 1;
  gfp.halfspace = false;
#ifndef WOODLAND_ACORN_HAVE_DC3D
  use_woodland_rg0c0 = true;
#endif
}

void ConvTest::init_yxz (const gallery::ZxyFn::Shape shape,
                         const Disloc::CPtr& disloc_,
                         const Real lx, const Real lz) {
  throw_if(lx <= 0 || lz <= 0, "init_yxz: lx and lz must be positive");
  init_zxy(shape, disloc_);
  use_yxz = true;
  yxz_lx = lx;
  yxz_lz = lz;
}

// Yudong 02/23/2026
int ConvTest::get_nx () const {
  return use_zxy ? nx_zxy : zxfn->get_nx();
}

int ConvTest::get_ny () const {
  return use_zxy ? ny_zxy : zxfn->get_ny();
}

void ConvTest::set_nx (const int nx) {
  assert(nx >= 1);
  if (use_zxy) {
    if (nx_zxy != nx) { t = nullptr; d = nullptr; }
    nx_zxy = nx;
    return;
  }
  if (zxfn->get_nx() != nx or zxfn->is_uniform_in_x() != xuniform) {
    t = nullptr; d = nullptr;
  }
  zxfn->set_nx(nx, xuniform);
}

void ConvTest::set_ny (const int ny) {
  assert(ny >= 1);
  if (use_zxy) {
    if (ny_zxy != ny) { t = nullptr; d = nullptr; }
    ny_zxy = ny;
    return;
  }
  if (zxfn->get_ny() != ny) { t = nullptr; d = nullptr; }
  zxfn->set_ny(ny);
}

// Yudong 02/23/2026
void ConvTest::set_nx_ny_zxy (const int nx, const int ny) {
  assert(use_zxy && nx >= 1 && ny >= 1);
  if (nx_zxy != nx || ny_zxy != ny) { t = nullptr; d = nullptr; }
  nx_zxy = nx;
  ny_zxy = ny;
}

void ConvTest::set_use_four_tris_per_rect(const bool use) {
  if ((use ? 4 : 2) != ntri_per_rect) { t = nullptr; d = nullptr; }
  ntri_per_rect = use ? 4 : 2;
}

void ConvTest::set_use_quad_cells (const bool use) {
  if (use != use_quad_cells) { t = nullptr; d = nullptr; }
  use_quad_cells = use;
}

void ConvTest::set_use_surface_recon (const bool use) {
  if (use != dargs.use_surface_recon) { t = nullptr; d = nullptr; }
  dargs.use_surface_recon = use;
}

void ConvTest::set_use_exact_tangents (const bool use) {
  if (use != dargs.use_exact_tangents) { t = nullptr; d = nullptr; }
  dargs.use_exact_tangents = use;
}

void ConvTest::set_use_c2_spline (const bool use) {
  if (use != dargs.use_c2_spline) { t = nullptr; d = nullptr; }
  dargs.use_c2_spline = use;
}

void ConvTest::set_use_halfspace (const bool use) {
  gfp.halfspace = use;
}

void ConvTest::set_tangent_recon_order (const int order) {
  if (order != dargs.tan_recon_order) { t = nullptr; d = nullptr; }
  dargs.tan_recon_order = order;
}

void ConvTest::set_disloc_order (const int order) {
  if (order != disloc_order) { t = nullptr; d = nullptr; }
  disloc_order = order;
}

void ConvTest::set_use_flat_elements (const bool use) {
  if (use != dargs.use_flat_elements) { t = nullptr; d = nullptr; }
  dargs.use_flat_elements = use;
}

void ConvTest::set_use_woodland_rg0c0 (const bool use) {
#ifndef WOODLAND_ACORN_HAVE_DC3D
  if (not use) {
    printf("WARNING: Using Woodland rg0c0 impl b/c dc3d impl is missing.\n");
    use_woodland_rg0c0 = true;
    return;
  }
#endif
  use_woodland_rg0c0 = use;
}

void ConvTest::set_use_nonunirect (const bool use) {
  if (use != use_nonunirect) { t = nullptr; d = nullptr; }
  use_nonunirect = use;
}

void ConvTest::set_xuniform (const bool use) {
  if (use_zxy) return;
  if (use != xuniform) { t = nullptr; d = nullptr; }
  xuniform = use;
  if (zxfn->get_nx() > 0) set_nx(zxfn->get_nx());
}

void ConvTest::set_support_req0 (const bool use) {
  if (use != dargs.support_req0) { t = nullptr; d = nullptr; }
  dargs.support_req0 = use;
}

void ConvTest::set_mesh_scale (const bool use) {
  if (use != dargs.mesh_scale) { t = nullptr; d = nullptr; }
  dargs.mesh_scale = use;
}

// Yudong 02/23/2026
static const char* zxy_shape_name (const gallery::ZxyFn::Shape s) {
  switch (s) {
  case gallery::ZxyFn::Shape::trig1_cosy: return "trig1_cosy";
  case gallery::ZxyFn::Shape::cos10_cos12_y: return "cos10_cos12_y";
  default: return "zxy";
  }
}

void ConvTest::print (FILE* fp) const {
  fprintf(fp, "ct> lam %1.3e mu %1.3e halfspace %d\n",
          gfp.lam, gfp.mu, int(gfp.halfspace));
  fprintf(fp, "ct> xuniform %d nonunirect %d ntriperrect %d dislocorder %d\n"
          "ct> exactsrf %d flatelem %d exacttan %d tanorder %d c2spline %d\n",
          int(xuniform), use_nonunirect, ntri_per_rect, disloc_order,
          int(not dargs.use_surface_recon), int(dargs.use_flat_elements),
          int(dargs.use_exact_tangents), dargs.tan_recon_order,
          int(dargs.use_c2_spline));
  if (use_yxz)
    fprintf(fp, "ct> yxz_shape %s nx %d nz %d lx %g lz %g\n",
            zxy_shape_name(zxy_shape), nx_zxy, ny_zxy, yxz_lx, yxz_lz);
  else if (use_zxy)
    fprintf(fp, "ct> zxy_shape %s nx %d ny %d lx %g ly %g\n",
            zxy_shape_name(zxy_shape), nx_zxy, ny_zxy, zxy_lx, zxy_ly);
  else
    fprintf(fp, "ct> zshape %s\n", ZxFn::convert(zxfn->get_shape()).c_str());
  if (use_woodland_rg0c0) printf("ct> using woodland impl of rg0c0\n");
  if (disloc)
    for (int i = 0; i < 3; ++i) {
      const auto& d = disloc->ds[i];
      const Real* const a = &d.amplitude;
      bool all0 = true;
      for (int k = 0; k < 9; ++k)
        if (a[k] != 0)
          all0 = false;
      if (all0) continue;
      fprintf(fp, "ct> disloc %d %s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f "
              "%6.3f %6.3f\n",
              i, Disloc::convert(d.shape).c_str(),
              a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    }
}

Triangulation::Ptr ConvTest
::triangulate (const ZxFn& zxfn, const int ntri_per_rect) {
  assert(zxfn.get_nx() >= 1 and zxfn.get_ny() >= 1);

  const auto t = std::make_shared<Triangulation>();
  t->begin_modifications();

  const auto nx = zxfn.get_nx(), ny = zxfn.get_ny(), pnx = nx+1,
    os = pnx*(ny+1);

  for (int yi = 0, pat = 0; yi < ny; ++yi) {
    for (int xi = 0; xi < nx; ++xi, ++pat) {
      if (ntri_per_rect == 4) {
        const int tris[][3] =
          {{pnx* yi    + xi+1, pnx*(yi+1) + xi+1, os + nx*yi + xi},
           {pnx*(yi+1) + xi  , pnx* yi    + xi  , os + nx*yi + xi},
           {pnx* yi    + xi  , pnx* yi    + xi+1, os + nx*yi + xi},
           {pnx*(yi+1) + xi+1, pnx*(yi+1) + xi  , os + nx*yi + xi}};
        for (int i = 0; i < 4; ++i) t->append_tri(tris[i]);
      } else {
        // Don't want the alternation: breaks symmetry and then can't use
        // fast-eval.
        if (1) { //(pat % 2 == 0) {
          const int tris[][3] =
            {{pnx*yi + xi, pnx*(yi+1) + xi+1, pnx*(yi+1) + xi  },
             {pnx*yi + xi, pnx* yi    + xi+1, pnx*(yi+1) + xi+1}};
          for (int i = 0; i < 2; ++i) t->append_tri(tris[i]);
        } else {
          const int tris[][3] =
            {{pnx*yi + xi,   pnx* yi    + xi+1, pnx*(yi+1) + xi},
             {pnx*yi + xi+1, pnx*(yi+1) + xi+1, pnx*(yi+1) + xi}};
          for (int i = 0; i < 2; ++i) t->append_tri(tris[i]);
        }
      }
    }
  }

  for (int yi = 0; yi <= ny; ++yi)
    for (int xi = 0; xi <= nx; ++xi)
      t->append_vtx();
  if (ntri_per_rect == 4)
    for (int yi = 0; yi < ny; ++yi)
      for (int xi = 0; xi < nx; ++xi)
        t->append_vtx();

  t->end_modifications();

  const auto xs = zxfn.get_xbs();
  const auto zs = zxfn.get_zbs();
  for (int xi = 0; xi <= nx; ++xi) {
    const auto x = xs[xi];
    const auto z = zs[xi];
    for (int yi = 0; yi <= ny; ++yi) {
      const Real y = Real(yi)/ny;
      Real* vtx = t->get_vtx(pnx*yi + xi);
      vtx[0] = x;
      vtx[1] = y;
      vtx[2] = z;
    }
  }
  if (ntri_per_rect == 4) {
    const auto zshape = zxfn.get_shape();
    for (int xi = 0; xi < nx; ++xi) {
      const auto x = (xs[xi] + xs[xi+1])/2;
      Real z, unused;
      gallery::eval(zshape, x, z, unused);
      for (int yi = 0; yi < ny; ++yi) {
        const Real y = (Real(yi)/ny + Real(yi+1)/ny)/2;
        Real* vtx = t->get_vtx(os + nx*yi + xi);
        vtx[0] = x;
        vtx[1] = y;
        vtx[2] = z;
      }
    }
  }

  return t;
}

// Yudong 02/20/2026
// Triangulate a 2D surface z = f(x,y) for (x,y) in [0,1]^2.
// nx and ny are the number of rectangles in the x and y directions.
// zxy_shape is the shape of the surface.
// ntri_per_rect is the number of triangles per rectangle.
// Returns a pointer to the triangulation.
Triangulation::Ptr ConvTest
::triangulate_zxy (const int nx, const int ny,
                   const gallery::ZxyFn::Shape zxy_shape,
                   const int ntri_per_rect,
                   const Real lx, const Real ly) {
  assert(nx >= 1 && ny >= 1 && lx > 0 && ly > 0);

  const auto t = std::make_shared<Triangulation>();
  t->begin_modifications();

  const int pnx = nx + 1;
  const int os = pnx * (ny + 1);
  // Build triangles.
  for (int yi = 0, pat = 0; yi < ny; ++yi) {
    for (int xi = 0; xi < nx; ++xi, ++pat) {
      if (ntri_per_rect == 4) {
        const int tris[][3] =
          {{pnx* yi    + xi+1, pnx*(yi+1) + xi+1, os + nx*yi + xi},
           {pnx*(yi+1) + xi  , pnx* yi    + xi  , os + nx*yi + xi},
           {pnx* yi    + xi  , pnx* yi    + xi+1, os + nx*yi + xi},
           {pnx*(yi+1) + xi+1, pnx*(yi+1) + xi  , os + nx*yi + xi}};
        for (int i = 0; i < 4; ++i) t->append_tri(tris[i]);
      } else {
        if (1) {
          const int tris[][3] =
            {{pnx*yi + xi, pnx*(yi+1) + xi+1, pnx*(yi+1) + xi  },
             {pnx*yi + xi, pnx* yi    + xi+1, pnx*(yi+1) + xi+1}};
          for (int i = 0; i < 2; ++i) t->append_tri(tris[i]);
        } else {
          const int tris[][3] =
            {{pnx*yi + xi,   pnx* yi    + xi+1, pnx*(yi+1) + xi},
             {pnx*yi + xi+1, pnx*(yi+1) + xi+1, pnx*(yi+1) + xi}};
          for (int i = 0; i < 2; ++i) t->append_tri(tris[i]);
        }
      }
    }
  }

  for (int yi = 0; yi <= ny; ++yi)
    for (int xi = 0; xi <= nx; ++xi)
      t->append_vtx();
  if (ntri_per_rect == 4)
    for (int yi = 0; yi < ny; ++yi)
      for (int xi = 0; xi < nx; ++xi)
        t->append_vtx();

  t->end_modifications();

  for (int xi = 0; xi <= nx; ++xi) {
    const Real u = Real(xi) / nx;
    const Real x = lx*u;
    for (int yi = 0; yi <= ny; ++yi) {
      const Real v = Real(yi) / ny;
      const Real y = ly*v;
      Real z;
      gallery::ZxyFn::eval(zxy_shape, u, v, z, nullptr, nullptr);
      Real* vtx = t->get_vtx(pnx*yi + xi);
      vtx[0] = x;
      vtx[1] = y;
      vtx[2] = z;
    }
  }
  if (ntri_per_rect == 4) {
    for (int xi = 0; xi < nx; ++xi) {
      const Real u = (xi + 0.5) / nx;
      const Real x = lx*u;
      for (int yi = 0; yi < ny; ++yi) {
        const Real v = (yi + 0.5) / ny;
        const Real y = ly*v;
        Real z;
        gallery::ZxyFn::eval(zxy_shape, u, v, z, nullptr, nullptr);
        Real* vtx = t->get_vtx(os + nx*yi + xi);
        vtx[0] = x;
        vtx[1] = y;
        vtx[2] = z;
      }
    }
  }

  return t;
}

Triangulation::Ptr ConvTest
::triangulate_yxz (const int nx, const int nz, const Real lx, const Real lz,
                   const gallery::ZxyFn::Shape shape,
                   const int ntri_per_rect) {
  throw_if(lx <= 0 || lz <= 0, "triangulate_yxz: lx and lz must be positive");
  auto t = triangulate_zxy(nx, nz, shape, ntri_per_rect);
  for (Idx vi = 0; vi < t->get_nvtx(); ++vi) {
    auto p = t->get_vtx(vi);
    const Real u = p[0], v = p[1];
    Real y;
    // ZxyFn's first coordinate is depth and its second is along strike. The
    // default trig1_cosy shape is independent of the second coordinate, so
    // xhat remains exactly global +x.
    gallery::ZxyFn::eval(shape, v, u, y, nullptr, nullptr);
    p[0] = lx*u;
    p[1] = y;
    p[2] = -lz*v;
  }
  return t;
}

namespace {

mesh::Mesh::CPtr make_quad_mesh_zxy_or_yxz (
  const int nx, const int nsecond, const Real lx, const Real lsecond,
  const gallery::ZxyFn::Shape shape, const bool use_yxz)
{
  throw_if(nx < 1 || nsecond < 1,
           "make_quad_mesh: grid dimensions must be positive");
  throw_if(lx <= 0 || lsecond <= 0,
           "make_quad_mesh: physical dimensions must be positive");

  const auto vtxs = mesh::make_vertices(3);
  const auto topo = mesh::make_topology();

  for (int j = 0; j <= nsecond; ++j) {
    const Real v = Real(j)/nsecond;
    for (int i = 0; i <= nx; ++i) {
      const Real u = Real(i)/nx;
      Real f;
      gallery::ZxyFn::eval(shape, use_yxz ? v : u, use_yxz ? u : v,
                           f, nullptr, nullptr);
      const Real p_zxy[3] = {lx*u, lsecond*v, f};
      const Real p_yxz[3] = {lx*u, f, -lsecond*v};
      vtxs->append_vtx(use_yxz ? p_yxz : p_zxy);
    }
  }

  const auto vi = [nx] (const int i, const int j) -> Idx {
    return j*(nx + 1) + i;
  };
  for (int j = 0; j < nsecond; ++j)
    for (int i = 0; i < nx; ++i) {
      const Idx cell[4] = {
        vi(i, j), vi(i+1, j), vi(i+1, j+1), vi(i, j+1)
      };
      topo->append_cell(4, cell);
    }
  topo->end_modifications();

  const auto relations = mesh::make_relations(topo);
  return mesh::make_mesh(vtxs, topo, relations);
}

} // namespace

mesh::Mesh::CPtr ConvTest
::make_quad_mesh_zxy (const int nx, const int ny,
                      const gallery::ZxyFn::Shape shape,
                      const Real lx, const Real ly) {
  throw_if(lx <= 0 || ly <= 0,
           "make_quad_mesh_zxy: lx and ly must be positive");
  return make_quad_mesh_zxy_or_yxz(nx, ny, lx, ly, shape, false);
}

mesh::Mesh::CPtr ConvTest
::make_quad_mesh_yxz (const int nx, const int nz, const Real lx, const Real lz,
                      const gallery::ZxyFn::Shape shape) {
  return make_quad_mesh_zxy_or_yxz(nx, nz, lx, lz, shape, true);
}

namespace {
struct Param2DUserFn : public GlobalZSurface::UserFn {
  Param2DUserFn (const ZxFn::Shape shape_) : shape(shape_) {}
  
  void eval (const Real x, const Real y, Real& f, RPtr g) const override {
    Real gx;
    gallery::eval(shape, x, f, gx);
    if (g) {
      g[0] = gx;
      g[1] = 0;
    }
  }

  ZxFn::Shape shape;
};

// Yudong 02/23/2026
// UserFn for z = f(x,y): full gradient g = (zx, zy).
// it is a small adapter that tells GlobalZSurface how to evaluate z = f(x,y) and its gradient.
struct Param2DUserFnZxy : public GlobalZSurface::UserFn {
  Param2DUserFnZxy (const gallery::ZxyFn::Shape shape_,
                    const Real lx_, const Real ly_)
    : shape(shape_), lx(lx_), ly(ly_) {}

  void eval (const Real x, const Real y, Real& f, RPtr g) const override {
    gallery::ZxyFn::eval(shape, x/lx, y/ly, f,
                         g ? &g[0] : nullptr, g ? &g[1] : nullptr);
    if (g) {
      g[0] /= lx;
      g[1] /= ly;
    }
  }

  gallery::ZxyFn::Shape shape;
  Real lx, ly;
};

struct Param2DUserFnYxz : public GlobalYSurface::UserFn {
  Param2DUserFnYxz (const gallery::ZxyFn::Shape shape_,
                    const Real lx_, const Real lz_)
    : shape(shape_), lx(lx_), lz(lz_) {}

  void eval (const Real x, const Real z, Real& f, RPtr g) const override {
    Real fdepth, falong;
    gallery::ZxyFn::eval(shape, -z/lz, x/lx, f,
                         g ? &fdepth : nullptr, g ? &falong : nullptr);
    if (g) {
      g[0] = falong/lx;
      g[1] = -fdepth/lz;
    }
  }

  gallery::ZxyFn::Shape shape;
  Real lx, lz;
};
} // namespace

Discretization::Ptr ConvTest
::discretize (const mesh::Mesh::CPtr& m, const Triangulation::Ptr& t,
              const ZxFn::Shape shape, const DiscretizeArgs a) {
  Surface::CPtr s;
  if (a.use_surface_recon) {
    assert(t);
    if (a.use_flat_elements) {
      const Real primary[] = {1,0,0};
      s = std::make_shared<FlatElementSurface>(t, primary);
    } else {
      using Recon = ExtrudedCubicSplineSurface::Recon;
      const auto recon = (a.use_c2_spline ? Recon::c2 :
                          a.use_exact_tangents ? Recon::c1_tan_exact :
                          a.tan_recon_order == 2 ? Recon::c1_tan_2 :
                          Recon::c1_tan_4);
      assert(recon != Recon::c1_tan_4 || a.tan_recon_order == 4);
      s = std::make_shared<ExtrudedCubicSplineSurface>(shape, t, recon);
    }
  } else {
    const auto ufn = std::make_shared<Param2DUserFn>(shape);
    const auto srf = std::make_shared<GlobalZSurface>(
      m, ufn, a.support_req0, a.mesh_scale);
    s = srf;
  }
  return make_discretization(m, s);
}

// Yudong 02/23/2026
// z = f(x,y) mode: use ZxyFn and triangulate_zxy.
Discretization::Ptr ConvTest
::discretize (const mesh::Mesh::CPtr& m, const Triangulation::Ptr& t,
              const gallery::ZxyFn::Shape zxy_shape, const DiscretizeArgs a) {
  Surface::CPtr srf;
  if (a.use_flat_elements) {
    const Real primary[] = {1, 0, 0};
    srf = std::make_shared<FlatElementSurface>(t, primary);
  } else if (a.use_surface_recon) {
    throw_if(!m, "ctzx::discretize(zxy): srfrecon requires a mesh");
    const auto& topo = *m->get_topo();
    for (Idx ci = 0; ci < m->get_ncell(); ++ci)
      throw_if(topo.get_cell(ci).n != 4,
               "ctzx::discretize(zxy): bicubic srfrecon requires quad cells");
    srf = std::make_shared<BicubicBezierSurface>(
      m, zxy_shape, a.zxy_lx, a.zxy_ly);
  } else {
    const auto ufn = std::make_shared<Param2DUserFnZxy>(
      zxy_shape, a.zxy_lx, a.zxy_ly);
    srf = std::make_shared<GlobalZSurface>(
      m, ufn, a.support_req0, a.mesh_scale);
  }
  return make_discretization(m, srf);
}

Discretization::Ptr ConvTest
::discretize_yxz (const mesh::Mesh::CPtr& m,
                  const gallery::ZxyFn::Shape shape,
                  const Real lx, const Real lz, const DiscretizeArgs a) {
  throw_if(a.use_surface_recon || a.use_flat_elements,
           "discretize_yxz: reconstruction and flat elements are not supported");
  const auto ufn = std::make_shared<Param2DUserFnYxz>(shape, lx, lz);
  const auto srf = std::make_shared<GlobalYSurface>(
    m, ufn, a.support_req0, a.mesh_scale);
  return make_discretization(m, srf);
}

void ConvTest
::pywrite (const std::string& python_filename, const RealArray& dislocs,
           const RealArray& sigmas, const std::string& dict,
           const bool append) const {
  const auto& mesh = *d->get_mesh();
  assert(dislocs.empty() || 3*mesh.get_ncell() == int(dislocs.size()));
  assert(sigmas .empty() || 6*mesh.get_ncell() == int(sigmas .size()));
  assert(not (dislocs.empty() && sigmas.empty()));
  FILE* fp = fopen(python_filename.c_str(), append ? "a" : "w");
  if (not append) pywrite_header(fp);
  fprintf(fp, "%s = {}\n", dict.c_str());
  fprintf(fp, "%s['nx'] = %d\n", dict.c_str(), get_nx());
  if (t) {
    fprintf(fp, "%s['type'] = 'mesh'\n", dict.c_str());
    std::string tdi = dict + "['m']";
    fprintf(fp, "%s = {}\n", tdi.c_str());
    ctzx::pywrite(fp, tdi, *t);
  } else {
    fprintf(fp, "%s['type'] = 'mesh'\n", dict.c_str());
    std::string tdi = dict + "['m']";
    fprintf(fp, "%s = {}\n", tdi.c_str());
    ctzx::pywrite(fp, tdi, mesh);
  }
  std::string ddi = dict + "['disloc']";
  if (dislocs.empty())
    fprintf(fp, "%s = None\n", ddi.c_str());
  else
    pywrite_double_array(fp, ddi, dislocs.size()/3, 3, dislocs.data());
  std::string sdi = dict + "['sigma']";
  if (sigmas.empty())
    fprintf(fp, "%s = None\n", sdi.c_str());
  else
    pywrite_double_array(fp, sdi, sigmas.size()/6, 6, sigmas.data());
  fclose(fp);
}

void ConvTest
::pywrite_okada (const std::string& python_filename, const int nxr, const int nyr,
                 const RealArray& dislocs, const RealArray& sigmas,
                 const std::string& dict, const bool append) const {
  const int nrect = nxr*nyr;
  assert(dislocs.empty() || int(dislocs.size()) == 3*nrect);
  assert(sigmas .empty() || int(sigmas .size()) == 6*nrect);
  assert(not (dislocs.empty() && sigmas.empty()));
  FILE* fp = fopen(python_filename.c_str(), append ? "a" : "w");
  if (not append) pywrite_header(fp);
  fprintf(fp, "%s = {}\n", dict.c_str());
  fprintf(fp, "%s['type'] = 'rect'\n", dict.c_str());
  fprintf(fp, "%s['nx'] = %d\n", dict.c_str(), nxr);
  fprintf(fp, "%s['ny'] = %d\n", dict.c_str(), nyr);
  std::string ddi = dict + "['disloc']";
  if (dislocs.empty())
    fprintf(fp, "%s = None\n", ddi.c_str());
  else
    pywrite_double_array(fp, ddi, nrect, 3, dislocs.data());
  std::string sdi = dict + "['sigma']";
  if (sigmas.empty())
    fprintf(fp, "%s = None\n", sdi.c_str());
  else
    pywrite_double_array(fp, sdi, nrect, 6, sigmas.data());
  fclose(fp);
}

static void
pywrite_zxfn (const std::string& python_filename, const ZxFn& zxfn,
              const std::string& dict, const bool append) {
  FILE* fp = fopen(python_filename.c_str(), append ? "a" : "w");
  if (not append) pywrite_header(fp);
  fprintf(fp, "%s = {}\n", dict.c_str());
  fprintf(fp, "%s['type'] = 'zxfn'\n", dict.c_str());
  fprintf(fp, "%s['nx'] = %d\n", dict.c_str(), zxfn.get_nx());
  std::string ddi = dict + "['x']";
  pywrite_double_array(fp, ddi, 1, zxfn.get_nx()+1, zxfn.get_xbs());
  ddi = dict + "['z']";
  pywrite_double_array(fp, ddi, 1, zxfn.get_nx()+1, zxfn.get_zbs());
  fclose(fp);
}

// Refine alternating y-strips by one level to test convergence when using a
// nonuniform qmesh.
//   Cells are grouped by y-strip.
mesh::Mesh::CPtr ConvTest::make_nonunirect_mesh (const ZxFn& zxfn) {
  assert(zxfn.get_nx() >= 1 && zxfn.get_ny() >= 1);

  const bool uniform = true; // optionally uniform
  if (uniform)
    printf("NOTE, make_nonunirect_mesh: logically rectangular grid.\n");
  const auto nx = zxfn.get_nx(), ny = zxfn.get_ny();

  const auto v = mesh::make_vertices(3);
  const auto to = mesh::make_topology();
  mesh::BoolArray v2bdy;

  // Build topo and vtxs.
  {
    // Given (x,y), compute z and add the vertex (x,y,z) to the vertices.
    const auto zshape = zxfn.get_shape();
    const auto xs = zxfn.get_xbs();
    const auto addvtx =
      [&] (const Real x, const Real y, const bool on_bdy) -> Idx {
        Real z, unused;
        gallery::eval(zshape, x, z, unused);
        const Real p[] = {x, y, z};
        v2bdy.push_back(on_bdy);
        return v->append_vtx(p);
      };
    // Given grid integer coordinates (i,j), return the associated vertex.
    using std::make_pair;
    std::map<std::pair<Idx,Idx>,Idx> c2vi;
    const auto getvi = [&] (const Idx i, const Idx j) -> Idx {
      const auto p = make_pair(i,j);
      const auto it = c2vi.find(p);
      assert(it != c2vi.end());
      return it->second;
    };
    // Build grid.
    Idx cell[4];
    for (Idx xi = 0; xi < nx; ++xi) {
      for (Idx yi = 0; yi < ny; ++yi) {
        if (xi == 0) {
          if (yi == 0) c2vi[make_pair(0,0)] = addvtx(xs[0], 0, true);
          c2vi[make_pair(0,2*(yi+1))] = addvtx(xs[0], Real(yi+1)/ny, true);
        }
        if (yi == 0) c2vi[make_pair(2*(xi+1),0)] = addvtx(xs[xi+1], 0, true);
        c2vi[make_pair(2*(xi+1), 2*(yi+1))] =
          addvtx(xs[xi+1], Real(yi+1)/ny, xi+1 == nx or yi+1 == ny);
      }
      if (xi % 2 == 0 or uniform) {
        for (Idx yi = 0; yi < ny; ++yi) {
          cell[0] = getvi(2*xi, 2*yi);
          cell[1] = getvi(2*(xi+1), 2*yi);
          cell[2] = getvi(2*(xi+1), 2*(yi+1));
          cell[3] = getvi(2*xi, 2*(yi+1));
          to->append_cell(mesh::Cell(4, cell));
        }
      } else {
        const Real xl = xs[xi], xr = xs[xi+1], xm = (xl + xr)/2;
        for (Idx yi = 0; yi < ny; ++yi) {
          const auto ym = Real(2*yi+1)/(2*ny);
          c2vi[make_pair(2*xi+1, 2*yi)] = addvtx(xm, Real(yi)/ny, yi == 0);
          c2vi[make_pair(2*xi, 2*yi+1)] = addvtx(xl, ym, false);
          c2vi[make_pair(2*xi+1, 2*yi+1)] = addvtx(xm, ym, false);
          c2vi[make_pair(2*(xi+1), 2*yi+1)] = addvtx(xr, ym, false);
        }
        c2vi[make_pair(2*xi+1, 2*ny)] = addvtx(xm, 1, true);
        for (int k = 0; k < 2; ++k) {
          for (Idx yi = 0; yi < 2*ny; ++yi) {
            cell[0] = getvi(2*xi+k, yi);
            cell[1] = getvi(2*xi+k+1, yi);
            cell[2] = getvi(2*xi+k+1, yi+1);
            cell[3] = getvi(2*xi+k, yi+1);
            to->append_cell(mesh::Cell(4, cell));
          }
        }
      }
    }
    to->end_modifications();
  }

  const auto ncell = to->get_ncell(), nvtx = v->get_nvtx();

  // Assert various entity counts of the grid.
  assert(to->get_nvtx() == nvtx);
  assert(v2bdy.size() == static_cast<size_t>(nvtx));
#ifndef NDEBUG
  int nbdy = 0;
  for (const auto& e : v2bdy)
    if (e) ++nbdy;
#endif
  if (uniform) {
    assert(ncell == nx*ny);
    assert(nvtx == (nx+1)*(ny+1));
    assert(nbdy == 2*(nx + ny));
  } else {
    assert(ncell == (nx+1)/2*ny + 4*(nx/2)*ny);
    assert(nvtx == (nx+1)*(ny+1) + 4*(nx/2)*ny + nx/2);
    assert(nbdy == 2*(nx + ny) + 2*(nx/2));
  }

  // Make neighbor sets. In this grid, every nbr is a vtx nbr.
  std::vector<std::set<Idx>> nbrss(ncell); {
    std::vector<std::set<Idx>> vi2ci(nvtx);
    for (Idx ci = 0; ci < ncell; ++ci) {
      const auto c = to->get_cell(ci);
      for (int i = 0; i < c.n; ++i)
        vi2ci[c[i]].insert(ci);
    }
    for (Idx ci = 0; ci < ncell; ++ci) {
      const auto c = to->get_cell(ci);
      for (int i = 0; i < c.n; ++i) {
        const auto& cjs = vi2ci[c[i]];
        for (const auto& cj : cjs)
          if (cj != ci) nbrss[ci].insert(cj);
      }
    }
  }
  // Convert nbr sets to arrays.
  mesh::IdxArray c2csi(ncell+1), c2cs;
  c2csi[0] = 0;
  for (Idx ci = 0; ci < ncell; ++ci) {
    const auto& nbrs = nbrss[ci];
    const Idx nnbr = static_cast<Idx>(nbrs.size());
    c2csi[ci+1] = c2csi[ci] + nnbr;
    for (const auto& cj : nbrs)
      c2cs.push_back(cj);
  }

  const auto re = mesh::make_relations(to, c2csi, c2cs, v2bdy);
  return mesh::make_mesh(v, to, re);
}

void ConvTest::discretize () {
  mesh::Mesh::CPtr m;
  // Yudong 02/23/2026
  if (use_zxy) {
    throw_if(use_nonunirect, "zxy mode does not support nonunirect=1");
    if (use_quad_cells) {
      throw_if(dargs.use_flat_elements,
               "quadrilateral cells require flatelem=0");
      throw_if(use_yxz && dargs.use_surface_recon,
               "surface=yxz does not support srfrecon=1");
      t = nullptr;
      m = use_yxz ?
        make_quad_mesh_yxz(nx_zxy, ny_zxy, yxz_lx, yxz_lz, zxy_shape) :
        make_quad_mesh_zxy(nx_zxy, ny_zxy, zxy_shape, zxy_lx, zxy_ly);
    } else {
      throw_if(dargs.use_surface_recon,
               "zxy bicubic srfrecon requires element=quad (or ntri=1)");
      t = use_yxz ?
        triangulate_yxz(nx_zxy, ny_zxy, yxz_lx, yxz_lz,
                        zxy_shape, ntri_per_rect) :
        triangulate_zxy(nx_zxy, ny_zxy, zxy_shape, ntri_per_rect,
                        zxy_lx, zxy_ly);
      const auto tr = std::make_shared<TriangulationRelations>(t);
      m = tri2mesh(*t, tr);
    }
    d = use_yxz ? discretize_yxz(m, zxy_shape, yxz_lx, yxz_lz, dargs) :
                  discretize(m, t, zxy_shape, dargs);
  } else {
    if (not use_nonunirect) {
      t = triangulate(*zxfn, ntri_per_rect);
      const auto tr = std::make_shared<TriangulationRelations>(t);
      m = tri2mesh(*t, tr);
    } else {
      m = make_nonunirect_mesh(*zxfn);
      throw_if(dargs.use_surface_recon,
               "nonunirect=1 does not support srfrecon=1");
    }
    d = discretize(m, t, zxfn->get_shape(), dargs);
  }
  d->set_disloc_order(disloc_order);
#ifdef WOODLAND_HAVE_HMMVP
  hmats = hmat::make_hmatrices(d);
#endif
}

static void
setup (const int testcase, ZxFn::Shape& zshape, Disloc::Ptr& disloc,
       const Real r = 1,
       gallery::ZxyFn::Shape* zxy_shape_out = nullptr) {
  disloc = std::make_shared<Disloc>();
  const auto dshape = Disloc::Shape::stapered;
  switch (testcase) {
  case -1:
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(0, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 10: case 11: case 12:
    zshape = ZxFn::Shape::zero;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(testcase - 10, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 20: case 21: case 22:
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(testcase - 20, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 30: case 31: case 32:
    zshape = ZxFn::Shape::trig0;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(testcase - 30, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 40: case 41: case 42:
    zshape = ZxFn::Shape::steep;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(testcase - 40, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 50: case 51: case 52:
    // Uniform over (0,1)^2: constant amplitude in one component (dim 0, 1, or 2).
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(testcase - 50, Disloc::Shape::uniform, 1, 0.5, 0.5, 1, 0, 1, 1);
    break;
  case 24:
    // Yudong 03/11/2026: slip tangent to surface, in xz plane (1,0,dz/dx)/sqrt(1+(dz/dx)^2).
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set_slope_xz(gallery::ZxyFn::Shape::trig1_cosy, dshape, 1, 0.5, 0.5, 1, 0, r, r);
    break;
  case 54:
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set_slope_xz(gallery::ZxyFn::Shape::trig1_cosy, Disloc::Shape::uniform, 1, 0.5, 0.5, 1, 0, 1, 1);
    break;
  case 70: case 71: case 72:
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::cos10_cos12_y;
    disloc->set(testcase - 70, Disloc::Shape::uniform, 1, 0.5, 0.5, 1, 0, 1, 1);
    break;
  case 74:
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::cos10_cos12_y;
    disloc->set_slope_xz(gallery::ZxyFn::Shape::cos10_cos12_y, Disloc::Shape::uniform,
                         1, 0.5, 0.5, 1, 0, 1, 1);
    break;
  case 84:
    // Tanh-window magnitude in XY with slip tangent to z=f(x,y) in xz plane.
    zshape = ZxFn::Shape::trig1;
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::cos10_cos12_y;
    disloc->set_slope_xz(gallery::ZxyFn::Shape::cos10_cos12_y, Disloc::Shape::tanh_window,
                         1, 0.5, 0.5, 1, 0, 1, 1);
    break;
  case 0: case 1: case 2: case 3:
  default:
    zshape = (testcase == 0 ? ZxFn::Shape::trig1 :
              testcase == 1 ? ZxFn::Shape::zero :
              testcase == 2 ? ZxFn::Shape::trig0 :
              ZxFn::Shape::ramp);
    if (zxy_shape_out) *zxy_shape_out = gallery::ZxyFn::Shape::trig1_cosy;
    disloc->set(0, dshape,  0.7, 0.5, 0.5, 1, -0.4, 0.75, 0.65);
    disloc->set(1, dshape, -0.3, 0.4, 0.6, 1,    1, 0.7 , 0.4 );
    disloc->set(2, dshape,  0.1, 0.5, 0.5, 1,    0, 0.75, 0.75);
    break;
  }
  const bool skip_boundary_check = (testcase >= 50 && testcase <= 54) || testcase == 24
    || testcase >= 70;
  if (not skip_boundary_check && not disloc->is_boundary_zero(1e-12)) {
    printf("Disloc is not 0 on boundary; exiting.\n");
    exit(-1);
  }
}

static void
calc_errors (const RealArray& s_true, const RealArray& s, Real l2_err[6],
             Real li_err[6], Surface::CPtr srf = nullptr) {
  const int ncell = int(s_true.size()/6);
  assert(s.size() == s_true.size());
  const auto nthr = acorn::get_max_threads();
  const int naccum = 24;
  std::vector<Real> a(nthr*naccum, 0);
  Workspace w;
  ompparfor for (int ti = 0; ti < ncell; ++ti) {
    const auto tid = acorn::get_thread_num();
    auto ai = &a[tid*naccum];
    const auto area = srf ? cell_surface_area(w, *srf, ti) : 1;
    for (int i = 0; i < 6; ++i) {
      const auto den = s_true[6*ti+i];
      const auto diff = s[6*ti+i] - den;
      ai[4*i+0] += acorn::square(diff)*area;
      ai[4*i+1] += acorn::square(den)*area;
      ai[4*i+2] = std::max(ai[4*i+2], std::abs(diff));
      ai[4*i+3] = std::max(ai[4*i+3], std::abs(den));
    }
  }
  for (int i = 0; i < 6; ++i) {
    Real l2n = 0, l2d = 0, lin = 0, lid = 0;
    for (int tid = 0; tid < nthr; ++tid) {
      auto ai = &a[tid*naccum + 4*i];
      l2n += ai[0];
      l2d += ai[1];
      lin = std::max(lin, ai[2]);
      lid = std::max(lid, ai[3]);
    }
    if (l2d == 0) l2d = 1;
    if (lid == 0) lid = 1;
    l2_err[i] = std::sqrt(l2n/l2d);
    li_err[i] = lin/lid;
  }
}

static void print_errors (const Real l2_err[6], const Real l2ep[6],
                          const Real li_err[6], const Real liep[6],
                          const Real fac = 0) {
  assert(not (l2ep || liep) || fac > 0);
  printf("l2:");
  for (int i = 0; i < 6; ++i) {
    Real ooa = 0;
    if (l2ep && l2_err[i] != 0)
      ooa = -std::log(l2_err[i]/l2ep[i])/std::log(fac);
    printf(" %8.2e (%4.2f)", l2_err[i], ooa);
  }
  printf("\nli:");
  for (int i = 0; i < 6; ++i) {
    Real ooa = 0;
    if (liep && l2_err[i] != 0)
      ooa = -std::log(li_err[i]/liep[i])/std::log(fac);
    printf(" %8.2e (%4.2f)", li_err[i], ooa);
  }
  printf("\n");
}

struct ConvData {
  Discretization::CPtr d;
  int ny;
  RealArray dislocs, sigmas;
  ZxFn::CPtr zxfn;
};

namespace {

struct ConvTestSettings {
  int testcase = 0;
  bool use_four_tris_per_rect = false;
  bool use_surface_recon = true;
  bool use_flat_elements = false;
  bool use_exact_tangents = false;
  bool use_c2_spline = false;
  bool use_woodland_rg0c0 = false;
  bool use_halfspace = false;
  bool use_nonunirect = false;
  bool use_hmmvp = false;
  bool xuniform = false;
  int tan_recon_order = 4;
  int disloc_order = 2;
  int e_max = -1;
  Real r = 0.8;
  int ny_override = -1;

  void set(const std::string& params);
};

static bool in (const std::string sub, const std::string sup) {
  return sup.find(sub) != std::string::npos;
}

static void print_valid_pairs () {
  printf("Valid pairs:\n"
         "  testcase: int [0]\n"
         "  ntri: 2, 4 [2]\n"
         "  srfrecon: bool (0,1) [1]\n"
         "  flatelem: bool [0]\n"
         "  exacttan: bool [0]\n"
         "  tanorder: 2, 4 [4]\n"
         "  c2spline: bool [0]\n"
         "  dislocorder: 0, 1, 2, 3 [2]\n"
         "  nres: int > 0 [-1]\n"
         "  woodlandrect: bool [0]\n"
         "  halfspace: bool [0]\n"
         "  nonunirect: bool [0]\n"
         "  hmmvp: bool [0]\n"
         "  xuniform: bool [0]\n");
}

void ConvTestSettings::set (const std::string& params) {
  const auto toks = acorn::split(params, ",");
  for (const auto& t : toks) {
    const auto keyval = acorn::split(t, "=");
    if (keyval.size() != 2) {
      printf("Token error: %s\n", t.c_str());
      print_valid_pairs();
      exit(-1);
    }
    const auto& key = keyval[0];
    const auto& val = keyval[1];
    const int ival = std::stoi(val);
    if (in("testcase", key)) testcase = ival;
    else if (in("ntri", key)) use_four_tris_per_rect = ival == 4;
    else if (in("srfrecon", key)) use_surface_recon = ival;
    else if (in("flatelem", key)) use_flat_elements = ival;
    else if (in("exacttan", key)) use_exact_tangents = ival;
    else if (in("tanorder", key)) tan_recon_order = ival;
    else if (in("c2spline", key)) use_c2_spline = ival;
    else if (in("dislocorder", key)) disloc_order = ival;
    else if (in("nres", key)) e_max = ival - 1;
    else if (in("woodlandrect", key)) use_woodland_rg0c0 = ival;
    else if (in("halfspace", key)) use_halfspace = ival;
    else if (in("nonunirect", key)) use_nonunirect = ival;
    else if (in("hmmvp", key)) use_hmmvp = ival;
    else if (in("xuniform", key)) xuniform = ival;
    // Unadvertised debug option with unspecified behavior.
    else if (in("nyoverride", key)) ny_override = ival;
    else {
      printf("Unrecognized key-value pair: %s %s\n",
             keyval[0].c_str(), keyval[1].c_str());
      print_valid_pairs();
      exit(-1);
    }
  }
#ifndef WOODLAND_ACORN_HAVE_DC3D
  use_woodland_rg0c0 = true;
#endif
}

void set_settings (const ConvTestSettings& s, ConvTest& ct) {
  if (s.use_four_tris_per_rect and s.use_surface_recon
      and not s.use_flat_elements) {
    printf("ExtrudedCubicSplineSurface cannot be used with ntri=4");
    throw_if(true, "set_settings: incompatible settings");
  }
  ct.set_verbosity(1);
  ct.set_use_four_tris_per_rect(s.use_four_tris_per_rect);
  ct.set_use_surface_recon(s.use_surface_recon);
  ct.set_use_exact_tangents(s.use_exact_tangents);
  ct.set_use_c2_spline(s.use_c2_spline);
  ct.set_use_flat_elements(s.use_flat_elements);
  ct.set_use_woodland_rg0c0(s.use_woodland_rg0c0);
  ct.set_use_halfspace(s.use_halfspace);
  ct.set_use_nonunirect(s.use_nonunirect);
  ct.set_xuniform(s.xuniform);
  ct.set_tangent_recon_order(s.tan_recon_order);
  ct.set_disloc_order(s.disloc_order);
  ct.set_general_lam_mu();
}

} // namespace

static void conv_test_exact (const bool my_method, const ConvTestSettings& s) {
  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(s.testcase, zshape, disloc, s.r);
  int e_max = s.e_max;

  ConvTest ct;
  ct.init(zshape, disloc);
  set_settings(s, ct);
  ct.print();

  const auto method = (s.use_hmmvp ?
                       ConvTest::EvalMethod::direct_hmmvp :
                       ConvTest::EvalMethod::fast);
  if (s.use_hmmvp) printf("ct> use_hmmvp\n");

  const int nxfac = 10;
  int ny0 = 0;
  Real l2ep[6], liep[6];
  Real t_accum[2] = {0};
  for (int e = 0; e <= e_max; ++e) {
    const bool last = e == e_max;
    const bool output = last;
    const auto resfac = (1 << e);
    const int nx = nxfac*resfac;

    ct.set_nx(nx);
    if (e == 0) ny0 = ct.get_ny();
    const int ny = s.ny_override > 0 ? s.ny_override : ny0*resfac;
    printf("nx ny %d %d\n", nx, ny);
    RealArray dislocs, sigmas, esigmas;
    Real t0, t1, t2;
    if (my_method) {
      t0 = acorn::dbg::gettime();
      ct.eval(dislocs, sigmas, method);
      t1 = acorn::dbg::gettime();
      ct.eval_exact_at_cell_ctrs(esigmas);
      t2 = acorn::dbg::gettime();
    } else {
      int iny = ny;
      t0 = acorn::dbg::gettime();
      ct.eval_okada(nx, iny, dislocs, sigmas);
      t1 = acorn::dbg::gettime();
      assert(iny == ny);
      ct.eval_exact_at_rect_ctrs(nx, ny, esigmas);
      t2 = acorn::dbg::gettime();
    }

    t_accum[0] += t1 - t0;
    t_accum[1] += t2 - t1;
    printf("t: method %1.2e (%1.2e) exact %1.2e (%1.2e)\n",
           t1 - t0, t_accum[0], t2 - t1, t_accum[1]);

    Real l2_err[6], li_err[6];
    calc_errors(esigmas, sigmas, l2_err, li_err,
                my_method ? ct.get_discretization()->get_surface() : nullptr);
    print_errors(l2_err, e == 0 ? nullptr : l2ep,
                 li_err, e == 0 ? nullptr : liep, 2);
    acorn::copy(6, l2_err, l2ep);
    acorn::copy(6, li_err, liep);

    if (output) {
      if (my_method) {
        ct.pywrite("ctzx_wl_exact.py", dislocs, esigmas, "d", false);
        ct.pywrite("ctzx_wl.py", dislocs, sigmas, "d", false);
      } else {
        ZxFn ozxfn(ct.get_zxfn()->get_shape());
        ozxfn.set_nx(nx);
        ct.pywrite_okada("ctzx_rect.py", nx, ny, dislocs, sigmas, "d", false);
        pywrite_zxfn("ctzx_rect.py", ozxfn, "z", true);
        ct.pywrite_okada("ctzx_rect_exact.py", nx, ny, dislocs, esigmas, "d", false);
        pywrite_zxfn("ctzx_rect_exact.py", ozxfn, "z", true);
      }
    }
  }
}

void convtest_w_vs_e (const std::string& params) {
  ConvTestSettings s;
  s.e_max = 3;
  s.set(params);
  printf("convtest_w_vs_e %d\n", s.testcase);
  conv_test_exact(true, s);
}

void convtest_o_vs_e (const std::string& params) {
  ConvTestSettings s;
  s.e_max = 5;
  s.set(params);
  printf("convtest_o_vs_e %d\n", s.testcase);
  conv_test_exact(false, s);
}
// Yudong 02/23/2026
// stress calculation on z = f(x,y) surface (no exact reference).
void run_stress_zxy (const std::string& params) {
  int nx = 4, ny = 4, testcase = 20, ntri = 2, dislocorder = 2;
  bool srfrecon = false;
  bool flatelem = false;
  const auto toks = acorn::split(params, ",");
  for (const auto& t : toks) {
    const auto keyval = acorn::split(t, "=");
    if (keyval.size() != 2) continue;
    const auto& key = keyval[0];
    const auto& val = keyval[1];
    const int ival = std::stoi(val);
    if (key.find("nx") != std::string::npos) nx = ival;
    else if (key.find("ny") != std::string::npos) ny = ival;
    else if (key.find("testcase") != std::string::npos) testcase = ival;
    else if (key.find("ntri") != std::string::npos) ntri = ival;
    else if (key.find("srfrecon") != std::string::npos) srfrecon = ival != 0;
    else if (key.find("dislocorder") != std::string::npos) dislocorder = ival;
    else if (key.find("flatelem") != std::string::npos) flatelem = ival != 0;
  }
  if (srfrecon && !flatelem) {
    printf("run_stress_zxy: srfrecon=1 (spline recon) not supported for zxy; use flatelem=1 for flat elements.\n");
    throw std::runtime_error("run_stress_zxy: srfrecon=1 not supported for zxy");
  }
  ZxFn::Shape zshape_dummy;
  gallery::ZxyFn::Shape zxy_shape;
  Disloc::Ptr disloc;
  setup(testcase, zshape_dummy, disloc, 0.8, &zxy_shape);

  ConvTest ct;
  ct.init_zxy(zxy_shape, disloc);
  ConvTestSettings s;
  s.testcase = testcase;
  s.use_surface_recon = false;
  s.use_four_tris_per_rect = (ntri == 4);
   // Allow caller to choose dislocation interpolation order.
  s.disloc_order = dislocorder;
  s.use_flat_elements = flatelem;
  set_settings(s, ct);
  ct.set_nx_ny_zxy(nx, ny);

  printf("run_stress_zxy nx=%d ny=%d testcase=%d ntri=%d flatelem=%d dislocorder=%d\n",
         nx, ny, testcase, ntri, int(flatelem), dislocorder);
  ct.print();

  Real t0 = acorn::dbg::gettime();
  ct.discretize();
  Real t1 = acorn::dbg::gettime();
  printf("discretize et %1.3e\n", t1 - t0);

  RealArray dislocs, sigmas;
  t0 = acorn::dbg::gettime();
  ct.eval(dislocs, sigmas, ConvTest::EvalMethod::direct);
  t1 = acorn::dbg::gettime();
  printf("eval et %1.3e\n", t1 - t0);

  const int nc = int(sigmas.size() / 6);
  Real l2 = 0;
  for (int i = 0; i < 6*nc; ++i) {
    assert(std::isfinite(sigmas[i]));
    l2 += sigmas[i]*sigmas[i];
  }
  l2 = std::sqrt(l2 / (6*nc));
  printf("ncell %d sigma L2 %12.5e\n", nc, l2);

  ct.pywrite("ctzx_zxy_stress.py", dislocs, sigmas, "d", false);
  printf("wrote ctzx_zxy_stress.py\n");
}

// Yudong 03/04/2026
// Stress calculation on z = f(x,y) surface using the Exact reference.
// Similar to run_stress_zxy, but calls eval_exact_at_cell_ctrs to compute
// the Exact::calc_stress-based reference at cell centers. This routine does
// not compute the BEM stress sigmas; it is for inspecting the exact field.
void run_stress_zxy_exact (const std::string& params) {
  int nx = 4, ny = 4, testcase = 20, ntri = 2, dislocorder = 2;
  bool srfrecon = false;
  bool flatelem = false;
  const auto toks = acorn::split(params, ",");
  for (const auto& t : toks) {
    const auto keyval = acorn::split(t, "=");
    if (keyval.size() != 2) continue;
    const auto& key = keyval[0];
    const auto& val = keyval[1];
    const int ival = std::stoi(val);
    if (key.find("nx") != std::string::npos) nx = ival;
    else if (key.find("ny") != std::string::npos) ny = ival;
    else if (key.find("testcase") != std::string::npos) testcase = ival;
    else if (key.find("ntri") != std::string::npos) ntri = ival;
    else if (key.find("srfrecon") != std::string::npos) srfrecon = ival != 0;
    else if (key.find("dislocorder") != std::string::npos) dislocorder = ival;
    else if (key.find("flatelem") != std::string::npos) flatelem = ival != 0;
  }
  if (srfrecon && !flatelem) {
    printf("run_stress_zxy_exact: srfrecon=1 (spline recon) not supported for zxy; use flatelem=1 for flat elements.\n");
    throw std::runtime_error("run_stress_zxy_exact: srfrecon=1 not supported for zxy");
  }

  ZxFn::Shape zshape_dummy;
  gallery::ZxyFn::Shape zxy_shape;
  Disloc::Ptr disloc;
  setup(testcase, zshape_dummy, disloc, 0.8, &zxy_shape);

  ConvTest ct;
  ct.init_zxy(zxy_shape, disloc);
  ConvTestSettings s;
  s.testcase = testcase;
  s.use_surface_recon = false;
  s.use_four_tris_per_rect = (ntri == 4);
  s.disloc_order = dislocorder;
  s.use_flat_elements = flatelem;
  set_settings(s, ct);
  ct.set_nx_ny_zxy(nx, ny);

  printf("run_stress_zxy_exact nx=%d ny=%d testcase=%d ntri=%d flatelem=%d dislocorder=%d\n",
         nx, ny, testcase, ntri, int(flatelem), dislocorder);
  ct.print();

  Real t0 = acorn::dbg::gettime();
  ct.discretize();
  Real t1 = acorn::dbg::gettime();
  printf("discretize et %1.3e\n", t1 - t0);

  RealArray esigmas;
  t0 = acorn::dbg::gettime();
  ct.eval_exact_at_cell_ctrs(esigmas);
  t1 = acorn::dbg::gettime();
  printf("eval_exact_at_cell_ctrs et %1.3e\n", t1 - t0);

  const int nc = int(esigmas.size() / 6);
  Real l2_e = 0;
  for (int i = 0; i < 6*nc; ++i) {
    const Real se = esigmas[i];
    assert(std::isfinite(se));
    l2_e += se*se;
  }
  l2_e   = std::sqrt(l2_e   / (6*nc));
  printf("ncell %d sigma_exact_L2 %12.5e\n", nc, l2_e);

  // Fill dislocs at cell centers for Python output (same as eval).
  RealArray dislocs(3*nc);
  const auto& srf = *ct.get_discretization()->get_surface();
  for (Idx ci = 0; ci < nc; ++ci) {
    Real p[3];
    srf.cell_ctr_xyz(ci, p);
    disloc->eval(p, &dislocs[3*ci]);
  }
  ct.pywrite("ctzx_zxy_stress_exact_ref.py", dislocs, esigmas, "d", false);
  printf("wrote ctzx_zxy_stress_exact_ref.py\n");
}

static int check_errors (const RealArray& sd, const RealArray& sf,
                         const Real abs_tol = 2e-8, const Real rel_tol = 1e-6,
                         const int display_cnt = 100, const bool verbose = false);

namespace {

struct MatrixZxyOpts {
  std::string surface = "yxz";
  std::string element = "tri";
  int nx = 4;
  int ny = 4;
  int nz = 4;
  Real lx = 1;
  Real ly = 1;
  Real lz = 1;
  int testcase = 20;
  int ntri = 2;
  int dislocorder = 2;
  int disloc_comp = 0;
  bool flatelem = false;
  bool srfrecon = false;
  bool halfspace = false;
  bool verify = false;
  bool simple_output = false;
  std::string outfile;
};

static const int matrix_zxy_ncomp = 2;

static bool matrix_is_yxz (const MatrixZxyOpts& opts) {
  return opts.surface == "yxz";
}

static bool matrix_use_quads (const MatrixZxyOpts& opts) {
  return opts.element == "quad" || opts.ntri == 1;
}

static int matrix_zxy_comp (const MatrixZxyOpts& opts, const int i) {
  static const int yxz_comps[matrix_zxy_ncomp] = {1, 3};
  static const int zxy_comps[matrix_zxy_ncomp] = {2, 5};
  return (matrix_is_yxz(opts) ? yxz_comps : zxy_comps)[i];
}

static const char* matrix_zxy_comp_name (const MatrixZxyOpts& opts,
                                         const int i) {
  static const char* yxz_names[matrix_zxy_ncomp] = {"xy", "yy"};
  static const char* zxy_names[matrix_zxy_ncomp] = {"xz", "zz"};
  return (matrix_is_yxz(opts) ? yxz_names : zxy_names)[i];
}

static void write_matrix_zxy_py (FILE* fp, const MatrixZxyOpts& opts,
                                 const mesh::Mesh& mesh,
                                 const Surface& surface,
                                 const int nc, const RealArray& A,
                                 const RealArray& dislocs,
                                 const RealArray& sigmas_direct) {
  pywrite_header(fp);
  fprintf(fp, "d = {}\n");
  if (opts.simple_output) {
    fprintf(fp, "d['m'] = {}\n");
    pywrite(fp, "d['m']", mesh);
    pywrite_double_array(fp, "d['A_xz']", nc, nc, &A[0]);
    pywrite_double_array(fp, "d['A_zz']", nc, nc, &A[nc*nc]);
    return;
  }
  fprintf(fp, "d['type'] = 'mesh'\n");
  fprintf(fp, "d['nx'] = %d\n", opts.nx);
  if (matrix_is_yxz(opts)) {
    fprintf(fp, "d['nz'] = %d\n", opts.nz);
    fprintf(fp, "d['lx'] = %24.16e\n", opts.lx);
    fprintf(fp, "d['lz'] = %24.16e\n", opts.lz);
    fprintf(fp, "d['x_range'] = (0.0, d['lx'])\n");
    fprintf(fp, "d['z_range'] = (-d['lz'], 0.0)\n");
    fprintf(fp, "d['surface'] = 'yxz'\n");
    fprintf(fp, "d['surface_equation'] = 'y=f(x,z)'\n");
  } else {
    fprintf(fp, "d['ny'] = %d\n", opts.ny);
    fprintf(fp, "d['lx'] = %24.16e\n", opts.lx);
    fprintf(fp, "d['ly'] = %24.16e\n", opts.ly);
    fprintf(fp, "d['x_range'] = (0.0, d['lx'])\n");
    fprintf(fp, "d['y_range'] = (0.0, d['ly'])\n");
    fprintf(fp, "d['surface'] = 'zxy'\n");
    fprintf(fp, "d['surface_equation'] = 'z=f(x,y)'\n");
  }
  fprintf(fp, "d['halfspace'] = %s\n", opts.halfspace ? "True" : "False");
  fprintf(fp, "d['srfrecon'] = %s\n", opts.srfrecon ? "True" : "False");
  fprintf(fp, "d['surface_representation'] = '%s'\n",
          opts.srfrecon ? "bicubic_bezier" : "analytic");
  fprintf(fp, "d['element'] = '%s'\n",
          matrix_use_quads(opts) ? "quad" : "tri");
  fprintf(fp, "d['cells_per_rectangle'] = %d\n",
          matrix_use_quads(opts) ? 1 : opts.ntri);
  fprintf(fp, "d['coordinate_frame'] = 'receiver_local'\n");
  fprintf(fp, "d['lcs_primary'] = [1.0, 0.0, 0.0]\n");
  if (matrix_is_yxz(opts))
    fprintf(fp, "d['lcs_axes'] = ['along_x', 'down_depth', 'normal_+y']\n");
  else
    fprintf(fp, "d['lcs_axes'] = ['x_tangent', 'y_tangent', 'normal_+z']\n");
  fprintf(fp, "d['ncell'] = %d\n", nc);
  fprintf(fp, "d['disloc_comp'] = %d\n", opts.disloc_comp);
  fprintf(fp, "d['testcase'] = %d\n", opts.testcase);
  fprintf(fp, "d['m'] = {}\n");
  pywrite(fp, "d['m']", mesh);
  if (const auto* bs = dynamic_cast<const BicubicBezierSurface*>(&surface)) {
    const auto& ctrl = bs->get_control_values();
    pywrite_double_array(fp, "d['bezier_ctrl_z']", nc, 16, ctrl.data());
    fprintf(fp, "d['bezier_ctrl_layout'] = 'cell x (v*4+u)'\n");
    fprintf(fp, "d['bezier_data'] = ['f', 'f_x', 'f_y', 'f_xy']\n");
  }
  fprintf(fp, "d['matrix_components'] = ['%s', '%s']\n",
          matrix_zxy_comp_name(opts, 0), matrix_zxy_comp_name(opts, 1));
  for (int ai = 0; ai < matrix_zxy_ncomp; ++ai) {
    std::string var = std::string("d['A_") + matrix_zxy_comp_name(opts, ai) + "']";
    pywrite_double_array(fp, var, nc, nc, &A[ai*nc*nc]);
  }
  if (not dislocs.empty()) {
    std::string ddi = "d['disloc']";
    pywrite_double_array(fp, ddi, nc, 3, dislocs.data());
    fprintf(fp, "d['dislocs'] = d['disloc']\n");
  }
  if (not sigmas_direct.empty()) {
    std::string sdi = "d['sigma']";
    pywrite_double_array(fp, sdi, nc, 6, sigmas_direct.data());
    fprintf(fp, "d['sigma_direct'] = d['sigma']\n");
  }
}

static int check_matrix_errors (const RealArray& reference,
                                const RealArray& matrix_result,
                                const Real abs_tol, const Real rel_tol) {
  assert(reference.size() == matrix_result.size());
  int nerr = 0;
  Real max_abs = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const Real e = std::abs(matrix_result[i] - reference[i]);
    max_abs = std::max(max_abs, e);
    if (not std::isfinite(reference[i]) ||
        not std::isfinite(matrix_result[i]) ||
        (e > abs_tol && e > rel_tol*std::abs(reference[i]))) {
      if (nerr < 10)
        pr(puf(i/6) pu(i % 6) pu(reference[i]) pu(matrix_result[i])
           pu(matrix_result[i] - reference[i]));
      ++nerr;
    }
  }
  if (nerr) printf("matrix verify max_abs %1.3e\n", max_abs);
  return nerr;
}

static int build_and_write_matrix_zxy (const MatrixZxyOpts& opts,
                                       RealArray* A_out = nullptr) {
  int nerr = 0;
  throw_if(opts.surface != "yxz" && opts.surface != "zxy",
           "matrix: surface must be yxz or zxy");
  throw_if(opts.element != "tri" && opts.element != "quad",
           "matrix: element must be tri or quad");
  throw_if(!matrix_use_quads(opts) && opts.ntri != 2 && opts.ntri != 4,
           "matrix: ntri must be 1, 2, or 4");
  throw_if(opts.nx < 1 || (matrix_is_yxz(opts) ? opts.nz : opts.ny) < 1,
           "matrix: grid dimensions must be positive");
  if (matrix_is_yxz(opts))
    throw_if(opts.lx <= 0 || opts.lz <= 0,
             "matrix_yxz: lx and lz must be positive");
  else
    throw_if(opts.lx <= 0 || opts.ly <= 0,
             "matrix_zxy: lx and ly must be positive");
  throw_if(opts.disloc_comp < 0 || opts.disloc_comp > 2,
           "matrix_yxz: disloccomp must be 0, 1, or 2");
  throw_if(matrix_is_yxz(opts) && opts.flatelem,
           "matrix_yxz: flatelem=1 is not supported");
  throw_if(matrix_use_quads(opts) && opts.flatelem,
           "matrix: element=quad does not support flatelem=1");
  throw_if(opts.srfrecon && matrix_is_yxz(opts),
           "matrix: srfrecon=1 currently supports only surface=zxy");
  throw_if(opts.srfrecon && !matrix_use_quads(opts),
           "matrix: bicubic srfrecon requires element=quad (or ntri=1)");
  throw_if(opts.srfrecon && opts.flatelem,
           "matrix: srfrecon=1 and flatelem=1 are mutually exclusive");
  throw_if(opts.simple_output && matrix_is_yxz(opts),
           "matrix: simple_output=1 supports only surface=zxy (A_xz/A_zz)");

  ZxFn::Shape zshape_dummy;
  gallery::ZxyFn::Shape zxy_shape;
  Disloc::Ptr disloc;
  setup(opts.testcase, zshape_dummy, disloc, 0.8, &zxy_shape);

  ConvTest ct;
  if (matrix_is_yxz(opts))
    ct.init_yxz(zxy_shape, disloc, opts.lx, opts.lz);
  else
    ct.init_zxy(zxy_shape, disloc, opts.lx, opts.ly);
  ConvTestSettings s;
  s.testcase = opts.testcase;
  s.use_surface_recon = opts.srfrecon;
  s.use_four_tris_per_rect = (!matrix_use_quads(opts) && opts.ntri == 4);
  s.disloc_order = opts.dislocorder;
  s.use_flat_elements = opts.flatelem;
  set_settings(s, ct);
  ct.set_use_quad_cells(matrix_use_quads(opts));
  ct.set_nx_ny_zxy(opts.nx, matrix_is_yxz(opts) ? opts.nz : opts.ny);
  ct.set_verbosity(0);
  // Preserve the exact physical (x,z) parameterization; anisotropic UV
  // scaling is not needed for this matrix path.
  if (matrix_is_yxz(opts)) ct.set_mesh_scale(false);
  ct.set_use_halfspace(opts.halfspace);

  if (matrix_is_yxz(opts))
    printf("matrix surface=yxz element=%s nx=%d nz=%d lx=%g lz=%g testcase=%d ntri=%d srfrecon=%d halfspace=%d dislocorder=%d disloccomp=%d\n",
           matrix_use_quads(opts) ? "quad" : "tri",
           opts.nx, opts.nz, opts.lx, opts.lz, opts.testcase, opts.ntri,
           int(opts.srfrecon), int(opts.halfspace), opts.dislocorder,
           opts.disloc_comp);
  else
    printf("matrix surface=zxy element=%s nx=%d ny=%d lx=%g ly=%g testcase=%d ntri=%d srfrecon=%d halfspace=%d dislocorder=%d disloccomp=%d\n",
           matrix_use_quads(opts) ? "quad" : "tri",
           opts.nx, opts.ny, opts.lx, opts.ly, opts.testcase, opts.ntri,
           int(opts.srfrecon), int(opts.halfspace), opts.dislocorder,
           opts.disloc_comp);

  Real t0 = acorn::dbg::gettime();
  ct.discretize();
  Real t1 = acorn::dbg::gettime();
  printf("discretize et %1.3e\n", t1 - t0);

  const auto d = ct.get_discretization();
  const auto& gfp = ct.get_gfp();
  const auto& mesh = *d->get_mesh();
  const auto nc = mesh.get_ncell();

  if (nc > 1000)
    printf("matrix: ncell=%d (2*nc^2=%.2e doubles); large grids may be slow.\n",
           int(nc), double(matrix_zxy_ncomp)*nc*nc);

  Stress stress(d);
  Stress::Options o, onbr;
  set_standard_options(o, onbr);

  RealArray A(matrix_zxy_ncomp*nc*nc);
  t0 = acorn::dbg::gettime();
  ompparfor for (Idx si = 0; si < nc; ++si) {
    for (Idx ri = 0; ri < nc; ++ri) {
      Real s[6];
      stress.calc_matrix_entries(gfp, si, ri, opts.disloc_comp, s, o, onbr);
      for (int ai = 0; ai < matrix_zxy_ncomp; ++ai)
        A[ai*nc*nc + ri*nc + si] = s[matrix_zxy_comp(opts, ai)];
    }
  }
  t1 = acorn::dbg::gettime();
  printf("fill matrix et %1.3e\n", t1 - t0);
  if (A_out) *A_out = A;

  RealArray dd, sd;
  ct.eval(dd, sd, ConvTest::EvalMethod::direct);

  FILE* fp = fopen(opts.outfile.c_str(), "w");
  if (not fp) {
    printf("matrix: cannot open %s\n", opts.outfile.c_str());
    return 1;
  }
  write_matrix_zxy_py(fp, opts, mesh, *d->get_surface(), nc, A, dd, sd);
  fclose(fp);
  printf("wrote %s (ncell=%d)\n", opts.outfile.c_str(), int(nc));

  if (opts.verify) {
    RealArray sm(6*nc, 0);
    RealArray sd_selected(6*nc, 0);
    ompparfor for (Idx ri = 0; ri < nc; ++ri) {
      for (int ai = 0; ai < matrix_zxy_ncomp; ++ai) {
        const int comp = matrix_zxy_comp(opts, ai);
        Real accum = 0;
        for (Idx si = 0; si < nc; ++si)
          accum += A[ai*nc*nc + ri*nc + si] * dd[3*si + opts.disloc_comp];
        sm[6*ri + comp] = accum;
        sd_selected[6*ri + comp] = sd[6*ri + comp];
      }
    }
    nerr += check_matrix_errors(
      sd_selected, sm, 1e5*mv3::eps, 1e7*mv3::eps);
    if (nerr)
      printf("matrix surface=%s: verify failed (A*disloc vs direct eval)\n",
             opts.surface.c_str());
    else
      printf("matrix surface=%s: verify passed\n", opts.surface.c_str());
  }

  return nerr;
}

} // namespace

void run_matrix_zxy (const std::string& params) {
  MatrixZxyOpts opts;
  bool ny_set = false, nz_set = false;
  const auto toks = acorn::split(params, ",");
  for (const auto& t : toks) {
    const auto keyval = acorn::split(t, "=");
    if (keyval.size() != 2) continue;
    const auto& key = keyval[0];
    const auto& val = keyval[1];
    if (key == "out") {
      opts.outfile = val;
      continue;
    }
    if (key == "surface") opts.surface = val;
    else if (key == "element") opts.element = val;
    else if (key == "lx") opts.lx = std::stod(val);
    else if (key == "ly") opts.ly = std::stod(val);
    else if (key == "lz") opts.lz = std::stod(val);
    else {
      const int ival = std::stoi(val);
      if (key == "nx") opts.nx = ival;
      else if (key == "ny") { opts.ny = ival; ny_set = true; }
      else if (key == "nz") { opts.nz = ival; nz_set = true; }
      else if (key == "testcase") opts.testcase = ival;
      else if (key == "ntri") opts.ntri = ival;
      else if (key == "dislocorder") opts.dislocorder = ival;
      else if (key == "disloccomp") opts.disloc_comp = ival;
      else if (key == "flatelem") opts.flatelem = ival != 0;
      else if (key == "srfrecon") opts.srfrecon = ival != 0;
      else if (key == "halfspace") opts.halfspace = ival != 0;
      else if (key == "verify") opts.verify = ival != 0;
      else if (key == "simple_output") opts.simple_output = ival != 0;
      else throw std::runtime_error("run_matrix_zxy: unrecognized option " + key);
    }
  }
  // Backward compatibility with commands that used ny as the second grid
  // dimension before surface=yxz introduced the clearer nz spelling.
  if (matrix_is_yxz(opts) && ny_set && !nz_set) opts.nz = opts.ny;
  if (opts.outfile.empty())
    opts.outfile = matrix_is_yxz(opts) ? "yxz_A.py" : "zxy_A.py";
  build_and_write_matrix_zxy(opts);
}

void run_case (const std::string& params) {
  //unittest();

  const bool eval_tri = 1;
  const bool eval_oka = 0;
  const bool eval_tri_exact = 1;
  const auto okada_method = ConvTest::EvalMethod::fast;

  ConvTestSettings s;
  s.set(params);
  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(s.testcase, zshape, disloc, 1.0);

  ConvTest ct;
  ct.init(zshape, disloc);
  set_settings(s, ct);
  ct.set_nx(30);
  printf("nx %d\n", ct.get_nx());
  ct.set_verbosity(1);
  ct.print();

  RealArray tsigmas, tdislocs;
  if (eval_tri) {
    const auto t0 = acorn::dbg::gettime();
    ct.eval(tdislocs, tsigmas);
    const auto t1 = acorn::dbg::gettime();
    printf("ct.eval et %1.3e\n", t1 - t0);
    ct.pywrite("ctzx_tri.py", tdislocs, tsigmas, "d", false);
  } else {
    ct.discretize();
  }
  if (eval_tri_exact) {
    RealArray esigmas;
    const auto t0 = acorn::dbg::gettime();
    ct.eval_exact_at_cell_ctrs(esigmas);
    const auto t1 = acorn::dbg::gettime();
    printf("ct.eval_exact_at_cell_ctrs et %1.3e\n", t1 - t0);
    ct.pywrite("ctzx_tri_exact.py", tdislocs, esigmas, "d", false);
  }
  if (eval_oka) {
#if 1
    // nx has to be large to handle the geometry well.
    const int nx = 11*ct.get_nx();
    int ny = 5*ct.get_ny();
#else
    int nx = 100, ny = 40;
#endif
    pr("rect" pu(nx) pu(ny));
    RealArray idislocs, isigmas;
    const auto t0 = acorn::dbg::gettime();
    switch (okada_method) {
    case ConvTest::EvalMethod::direct: {
      RealArray dislocs, sigmas;
      ct.eval_okada(nx, ny, dislocs, sigmas, okada_method);
      ct.pywrite_okada("ctzx_rect.py", nx, ny, dislocs, sigmas, "d", false);
      ct.interp_okada(nx, ny, sigmas, isigmas);
    } break;
    case ConvTest::EvalMethod::fast: {
      ConvTest::SupportPoints supports;
      ct.collect_rect_support_points(nx, ny, supports);
      RealArray sigmas;
      ct.eval_okada(nx, ny, supports, sigmas);
      ct.interp_okada(nx, ny, supports, sigmas, isigmas);
    } break;
    default: assert(0);
    }
    const auto t1 = acorn::dbg::gettime();
    printf("ct.eval/interp_okada et %1.3e\n", t1 - t0);
    ct.pywrite("ctzx_tri_interp.py", idislocs, isigmas, "d", false);
    if (not (tsigmas.empty() || isigmas.empty())) {
      Real l2_err[6], li_err[6];
      calc_errors(isigmas, tsigmas, l2_err, li_err,
                  ct.get_discretization()->get_surface());
      print_errors(l2_err, nullptr, li_err, nullptr);
    }
  }
}

static Real testfn(const int fno, const Real x, const Real y) {
  switch (fno) {
  case 0:
  default:
    return std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
  }
}

void set_standard_options (Stress::Options& o, Stress::Options& o_nbr) {
  // Use nbr relations rather than geometry to determine when to call calc_hfp
  // on other-interactions.
  o.near_dist_fac = 0;
  o_nbr = o;
  if (use_calc_integral_tensor_quadrature) {
    o_nbr.qp_tq.np_radial = 40;
    o_nbr.qp_tq.np_angular = 40;
    o_nbr.force_tensor_quadrature = true;
  } else {
    o_nbr.force_hfp = true;
  }
}

int ConvTest::test_interp () {
  const bool verbose = false;
  int nerr = 0;

  ZxFn zxfn(ZxFn::Shape::trig1), zxfnr(zxfn.get_shape());
  const auto disloc = std::make_shared<Disloc>();
  ConvTest ct;
  ct.init(zxfn.get_shape(), disloc);
  const int fno = 0;

  Real pl2e = 0, plie = 0;
  for (int e = 0; e < 6; ++e) {
    const int nx = 5*(1 << e);
    zxfn.set_nx(nx);
    const int ny = zxfn.get_ny();
    const int onx = std::ceil(3*nx), ony = std::ceil(1.6*ny);
    ct.set_nx(nx);
    ct.discretize();

    RealArray tfs; {
      RealArray rfs(6*ony*onx, 0);
      zxfnr.set_nx(onx);
      const auto rxs = zxfnr.get_xbs();
      for (int iy = 0, k = 0; iy < ony; ++iy) {
        const auto y = (iy + 0.5)/ony;
        for (int ix = 0; ix < onx; ++ix, ++k) {
          const auto x = (rxs[ix] + rxs[ix+1])/2;
          rfs[6*k] = testfn(fno, x, y);
        }
      }
      ct.interp_okada(onx, ony, rfs, tfs);
    }

    const auto& m = *ct.d->get_mesh();
    const auto& srf = *ct.d->get_surface();
    const auto ntri = m.get_ncell();
    RealArray tfs_true(ntri);
    for (int ti = 0; ti < ntri; ++ti) {
      Real p[3];
      srf.cell_ctr_xyz(ti, p);
      tfs_true[ti] = testfn(fno, p[0], p[1]);
    }

    Real l2n = 0, l2d = 0, lin = 0, lid = 0;
    for (int ti = 0; ti < ntri; ++ti) {
      const auto d = tfs[6*ti] - tfs_true[ti];
      l2n += acorn::square(d);
      l2d += acorn::square(tfs_true[ti]);
      lin = std::max(lin, std::abs(d));
      lid = std::max(lid, std::abs(tfs_true[ti]));
    }
    const auto l2e = std::sqrt(l2n/l2d), lie = lin/lid;
    const auto l2_ooa = e == 0 ? 0 : std::log(pl2e/l2e)/std::log(2);
    const auto li_ooa = e == 0 ? 0 : std::log(plie/lie)/std::log(2);

    if (e > 4) {
      if (l2_ooa < 1.98) ++nerr;
      if (li_ooa < 1.95) ++nerr;
    }
    if (verbose)
      printf("%2d nx %3d ny %3d l2 %9.3e (%5.3f) linf %9.3e (%5.3f)\n",
             e, nx, ny, l2e, l2_ooa, lie, li_ooa);

    pl2e = l2e;
    plie = lie;
  }
  if (nerr) printf("ctzx: test_interp failed\n");
  return nerr;
}

static int test_fast_okada_methods (const bool use_own_impl) {
  int nerr = 0;

  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(0, zshape, disloc);

  ConvTest ct;
  ct.init(zshape, disloc);
  ct.set_verbosity(0);
  const int nx = use_own_impl ? 7 : 11;
  ct.set_nx(nx);
  ct.discretize();
  const int ny = ct.get_zxfn()->get_ny();
  ct.set_use_woodland_rg0c0(use_own_impl);

  const int onx = 4*nx;
  int ony = ny + 9;

  RealArray s2, is2; {
    RealArray dislocs, s1;
    ct.eval_okada(onx, ony, dislocs, s1, ConvTest::EvalMethod::direct);
    ct.eval_okada(onx, ony, dislocs, s2);
    const int n = int(s1.size());
    for (int i = 0; i < n; ++i) {
      const auto e = std::abs(s2[i] - s1[i]);
      if (e > (use_own_impl ? 1e-7 : 1e-9) ||
          e > (use_own_impl ? 1e-5 : 1e-6)*std::abs(s1[i])) {
        if (nerr < 10) pr(puf(i) pu(s1[i]) pu(s2[i]) pu(s2[i] - s1[i]));
        ++nerr;
      }
    }
    ct.interp_okada(onx, ony, s2, is2);
  }

  {
    ConvTest::SupportPoints supports;
    ct.collect_rect_support_points(onx, ony, supports);
    RealArray s3, is3;
    ct.eval_okada(onx, ony, supports, s3);
    const int ns = int(supports.size());
    for (int i = 0; i < ns; ++i) {
      const auto& sp = supports[i];
      const int idx = sp.iy*onx + sp.ix;
      for (int k = 0; k < 6; ++k) {
        const int i3 = 6*i + k, i2 = 6*idx + k;
        if (s3[i3] != s2[i2]) {
          if (nerr < 10)
            pr(puf(i) pu(idx) pu(k) pu(s2[i2]) pu(s3[i3]) pu(s3[i3] - s2[i2]));
          ++nerr;
        }
      }
    }
    ct.interp_okada(onx, ony, supports, s3, is3);
    const int n = int(is2.size());
    for (int i = 0; i < n; ++i)
      if (is3[i] != is2[i]) {
        if (nerr < 10) pr(puf(i) pu(is2[i]) pu(is3[i]) pu(is3[i] - is2[i]));
        ++nerr;
      }
  }

  if (nerr) printf("ctzx: test_fast_okada_methods failed\n");
  return nerr;
}

static int
check_errors (const RealArray& sd, const RealArray& sf,
              const Real abs_tol, const Real rel_tol,
              const int display_cnt, const bool verbose) {
  int nerr = 0;
  const auto nc = sd.size() / 6;
  Real max_abs = 0;
  for (size_t i = 0; i < 6*nc; ++i) {
    const Real e = std::abs(sf[i] - sd[i]);
    max_abs = std::max(max_abs, e);
    if (std::isnan(sf[i]) or std::isnan(sd[i]) or e > abs_tol or
        e > rel_tol*std::abs(sd[i])) {
      if (nerr < display_cnt)
        pr(puf(i/6) pu(i % 6) pu(sd[i]) pu(sf[i]) pu(sf[i] - sd[i]));
      ++nerr;
    }
  }
  if (nerr) printf("check_errors max_abs %1.3e\n", max_abs);
  return nerr;
}

static int test_fast_woodland_methods (const bool use_nonunirect = false) {
  int nerr = 0;

  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(0, zshape, disloc);

  ConvTest ct;
  ct.init(zshape, disloc);
  ct.set_verbosity(0);
  ct.set_use_surface_recon(0);
  ct.set_use_exact_tangents(1);
  ct.set_disloc_order(2);
  ct.set_use_nonunirect(use_nonunirect);

  for (const bool use_four : {false, true}) {
    if (use_nonunirect and use_four) continue;
    ct.set_use_four_tris_per_rect(use_four);
    ct.set_nx(use_four ? 5 : 6);
    ct.set_use_halfspace(not use_four);

    RealArray dd, sd, df, sf, se;
    const auto t0 = acorn::dbg::gettime();
    ct.eval(dd, sd, ConvTest::EvalMethod::direct);
    const auto t1 = acorn::dbg::gettime();
    ct.eval(df, sf, ConvTest::EvalMethod::fast);
    const auto t2 = acorn::dbg::gettime();
    if (0) {
      // We don't expect speedup for a test-problem size. Each disloc component
      // requires 2*6 GF evaluations (2 for each side of the source, 6 for the 6
      // components of the quadratic fit), so a fully general problem requires
      // 2*6*3 = 36 GF evaluations. Thus, breakeven is roughly nyrect = 36. For a
      // single disloc component, breakeven is 12.
      printf("eval_direct/fast et %1.3e %1.3e speedup %1.2f\n",
             t1 - t0, t2 - t1, (t1 - t0)/(t2 - t1));
    }
    ct.pywrite(use_nonunirect ? "ctzx_qmesh.py" : "ctzx_tri.py",
               df, sf, "d", false);
    nerr += check_errors(sd, sf);
  }

  if (nerr) printf("ctzx: test_fast_woodland_methods failed\n");
  return nerr;
}

static int test_fast_nonunirect () {
  return test_fast_woodland_methods(true);
}

static int test_matrix_form () {
  int nerr = 0;

  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(20, zshape, disloc);
  const int disloc_comp = 0;

  ConvTest ct;
  ct.init(zshape, disloc);
  ct.set_verbosity(0);
  ct.set_use_surface_recon(0);
  ct.set_disloc_order(2);
  ct.set_nx(4);
  ct.set_use_halfspace(false);
  ct.discretize();

  const auto d = ct.get_discretization();
  const auto& gfp = ct.get_gfp();

  Stress stress(d);

  Stress::Options o, onbr;
  set_standard_options(o, onbr);

  // Form 6 nc x nc matrices.
  const auto& m = *d->get_mesh();
  const auto nc = m.get_ncell();
  RealArray A(6*nc*nc);
  ompparfor for (Idx si = 0; si < nc; ++si) {
    for (Idx ri = 0; ri < nc; ++ri) {
      Real s[6];
      stress.calc_matrix_entries(gfp, si, ri, disloc_comp, s, o, onbr);
      for (int i = 0; i < 6; ++i)
        A[i*nc*nc + ri*nc + si] = s[i];
    }
  }

  RealArray dd, sd;
  ct.eval(dd, sd, ConvTest::EvalMethod::direct);

  // 6 matvecs.
  RealArray sm(6*nc, 0);
  ompparfor for (Idx ri = 0; ri < nc; ++ri) {
    for (int comp = 0; comp < 6; ++comp) {
      Real accum = 0;
      for (Idx si = 0; si < nc; ++si)
        accum += A[comp*nc*nc + ri*nc + si] * dd[3*si + disloc_comp];
      sm[6*ri + comp] = accum;
    }
  }

  nerr += check_errors(sd, sm, 1e5*mv3::eps, 1e7*mv3::eps);

  return nerr;
}

static int test_yxz_geometry () {
  int nerr = 0;
  const int nx = 3, nz = 2;
  const Real lx = 2, lz = 1;
  const auto shape = gallery::ZxyFn::Shape::trig1_cosy;
  const auto t = ConvTest::triangulate_yxz(nx, nz, lx, lz, shape, 2);

  Real xmin = lx, xmax = 0, zmin = 0, zmax = -lz;
  for (Idx vi = 0; vi < t->get_nvtx(); ++vi) {
    const auto p = t->get_vtx(vi);
    xmin = std::min(xmin, p[0]); xmax = std::max(xmax, p[0]);
    zmin = std::min(zmin, p[2]); zmax = std::max(zmax, p[2]);
  }
  const Real tol = 1e4*mv3::eps;
  if (std::abs(xmin) > tol || std::abs(xmax-lx) > tol ||
      std::abs(zmin+lz) > tol || std::abs(zmax) > tol) {
    printf("yxz geometry ranges: x [%g,%g], z [%g,%g]\n",
           xmin, xmax, zmin, zmax);
    ++nerr;
  }

  const auto tr = std::make_shared<TriangulationRelations>(t);
  const auto m = tri2mesh(*t, tr);
  ConvTest::DiscretizeArgs a;
  a.use_surface_recon = false;
  a.mesh_scale = false;
  const auto d = ConvTest::discretize_yxz(m, shape, lx, lz, a);
  const auto& srf = *d->get_surface();
  for (Idx ci = 0; ci < srf.get_ncell(); ++ci) {
    Real p[3], lcs[9];
    srf.cell_ctr_xyz(ci, p);
    srf.cell_ctr_lcs(ci, lcs);
    if (not (p[0] > 0 && p[0] < lx && p[2] > -lz && p[2] < 0)) {
      printf("yxz geometry cell center outside open domain\n");
      ++nerr;
      break;
    }
    if (std::abs(lcs[0]-1) > tol || std::abs(lcs[1]) > tol ||
        std::abs(lcs[2]) > tol || lcs[7] <= 0) {
      printf("yxz geometry unexpected LCS xhat or normal orientation\n");
      ++nerr;
      break;
    }
  }
  return nerr;
}

static int test_zxy_scaled_geometry () {
  int nerr = 0;
  const int nx = 3, ny = 2;
  const Real lx = 8, ly = 5;
  const auto shape = gallery::ZxyFn::Shape::cos10_cos12_y;
  const Real tol = 1e4*mv3::eps;

  // Triangle vertices must span the requested physical rectangle while
  // retaining the same normalized testcase shape and height amplitude.
  const auto t = ConvTest::triangulate_zxy(nx, ny, shape, 4, lx, ly);
  Real xmin = lx, xmax = 0, ymin = ly, ymax = 0;
  for (Idx vi = 0; vi < t->get_nvtx(); ++vi) {
    const auto p = t->get_vtx(vi);
    xmin = std::min(xmin, p[0]); xmax = std::max(xmax, p[0]);
    ymin = std::min(ymin, p[1]); ymax = std::max(ymax, p[1]);
    Real z;
    gallery::ZxyFn::eval(shape, p[0]/lx, p[1]/ly,
                         z, nullptr, nullptr);
    if (std::abs(p[2] - z) > tol) {
      printf("scaled zxy triangle vertex is off normalized surface\n");
      ++nerr;
      break;
    }
  }
  if (std::abs(xmin) > tol || std::abs(xmax-lx) > tol ||
      std::abs(ymin) > tol || std::abs(ymax-ly) > tol) {
    printf("scaled zxy ranges: x [%g,%g], y [%g,%g]\n",
           xmin, xmax, ymin, ymax);
    ++nerr;
  }

  // The analytic surface adapter must return physical-coordinate gradients.
  const auto m = ConvTest::make_quad_mesh_zxy(nx, ny, shape, lx, ly);
  ConvTest::DiscretizeArgs a;
  a.use_surface_recon = false;
  a.mesh_scale = false;
  a.zxy_lx = lx;
  a.zxy_ly = ly;
  const auto d = ConvTest::discretize(m, nullptr, shape, a);
  const auto& srf = *d->get_surface();
  for (Idx ci = 0; ci < srf.get_ncell(); ++ci) {
    Real p[3], lcs[9], z, zu, zv;
    srf.cell_ctr_xyz(ci, p);
    srf.cell_ctr_lcs(ci, lcs);
    gallery::ZxyFn::eval(shape, p[0]/lx, p[1]/ly, z, &zu, &zv);
    Real n[] = {-zu/lx, -zv/ly, 1};
    mv3::normalize(n);
    if (std::abs(p[2]-z) > tol || mv3::distance2(&lcs[6], n) > tol) {
      printf("scaled zxy analytic surface or gradient mismatch\n");
      ++nerr;
      break;
    }
  }

  // A scaled bicubic patch must match its unit-domain counterpart in z;
  // its physical z derivatives differ by 1/lx and 1/ly.
  const auto mu = ConvTest::make_quad_mesh_zxy(nx, ny, shape);
  BicubicBezierSurface su(mu, shape);
  BicubicBezierSurface ss(m, shape, lx, ly);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const Idx ci = j*nx+i;
      const Real u = (i + Real(0.37))/nx;
      const Real v = (j + Real(0.61))/ny;
      const Real uvu[] = {u, v};
      const Real uvs[] = {lx*u, ly*v};
      Real pu[3], ps[3], Ju[6], Js[6];
      su.cell_position1(ci, uvu, pu, nullptr, nullptr, Ju);
      ss.cell_position1(ci, uvs, ps, nullptr, nullptr, Js);
      const Real err = std::max({
        std::abs(ps[0]-lx*pu[0]), std::abs(ps[1]-ly*pu[1]),
        std::abs(ps[2]-pu[2]), std::abs(Js[4]-Ju[4]/lx),
        std::abs(Js[5]-Ju[5]/ly)});
      if (err > tol) {
        printf("scaled zxy bicubic surface mismatch: %1.3e\n", err);
        ++nerr;
        return nerr;
      }
    }
  return nerr;
}

static int test_quad_mesh_zxy_yxz () {
  int nerr = 0;
  const int nx = 3, nsecond = 2;
  const Real lx = 2, lz = 1;
  const auto shape = gallery::ZxyFn::Shape::trig1_cosy;
  for (const bool yxz : {false, true}) {
    const auto m = yxz ?
      ConvTest::make_quad_mesh_yxz(nx, nsecond, lx, lz, shape) :
      ConvTest::make_quad_mesh_zxy(nx, nsecond, shape);
    const auto& to = *m->get_topo();
    const auto& vtxs = *m->get_vtxs();
    if (m->get_ncell() != nx*nsecond ||
        vtxs.get_nvtx() != (nx+1)*(nsecond+1)) {
      printf("quad mesh %s: wrong entity counts\n", yxz ? "yxz" : "zxy");
      ++nerr;
    }
    for (Idx ci = 0; ci < m->get_ncell(); ++ci)
      if (to.get_cell(ci).n != 4) {
        printf("quad mesh %s: cell %d does not have four vertices\n",
               yxz ? "yxz" : "zxy", int(ci));
        ++nerr;
        break;
      }

    ConvTest::DiscretizeArgs a;
    a.use_surface_recon = false;
    a.mesh_scale = false;
    const auto d = yxz ?
      ConvTest::discretize_yxz(m, shape, lx, lz, a) :
      ConvTest::discretize(m, nullptr, shape, a);
    const auto& srf = *d->get_surface();
    for (Idx ci = 0; ci < srf.get_ncell(); ++ci) {
      Real p[3], f;
      srf.cell_ctr_xyz(ci, p);
      if (yxz)
        gallery::ZxyFn::eval(shape, -p[2]/lz, p[0]/lx, f,
                             nullptr, nullptr);
      else
        gallery::ZxyFn::eval(shape, p[0], p[1], f, nullptr, nullptr);
      const Real actual = yxz ? p[1] : p[2];
      if (std::abs(actual - f) > 1e4*mv3::eps) {
        printf("quad mesh %s: cell center is off the analytic surface\n",
               yxz ? "yxz" : "zxy");
        ++nerr;
        break;
      }
    }
  }
  return nerr;
}

static int test_matrix_form_zxy () {
  int nerr = 0;
  RealArray A_full, A_half;
  for (const bool halfspace : {false, true}) {
    MatrixZxyOpts opts;
    opts.surface = "yxz";
    opts.nx = 3;
    opts.nz = 3;
    opts.lx = 2;
    opts.lz = 1;
    opts.testcase = 20;
    opts.disloc_comp = 0;
    opts.halfspace = halfspace;
    opts.verify = true;
    opts.outfile = halfspace ? "/tmp/ctzx_yxz_matrix_halfspace_unittest.py" :
                               "/tmp/ctzx_yxz_matrix_fullspace_unittest.py";
    nerr += build_and_write_matrix_zxy(opts, halfspace ? &A_half : &A_full);
  }
  if (A_full.size() != A_half.size() || A_full.empty()) {
    printf("matrix_yxz halfspace test: incompatible matrix sizes\n");
    ++nerr;
  } else {
    Real max_diff = 0;
    for (size_t i = 0; i < A_full.size(); ++i) {
      if (not std::isfinite(A_full[i]) || not std::isfinite(A_half[i])) {
        printf("matrix_yxz halfspace test: nonfinite matrix entry\n");
        ++nerr;
        break;
      }
      max_diff = std::max(max_diff, std::abs(A_full[i] - A_half[i]));
    }
    if (max_diff <= 1e3*mv3::eps) {
      printf("matrix_yxz halfspace test: halfspace had no measurable effect\n");
      ++nerr;
    }
  }

  RealArray A_zxy_full, A_zxy_half;
  for (const bool halfspace : {false, true}) {
    MatrixZxyOpts opts;
    opts.surface = "zxy";
    opts.nx = 3;
    opts.ny = 3;
    opts.testcase = 20;
    opts.disloc_comp = 0;
    opts.halfspace = halfspace;
    opts.verify = true;
    opts.outfile = halfspace ? "/tmp/ctzx_zxy_matrix_halfspace_unittest.py" :
                               "/tmp/ctzx_zxy_matrix_fullspace_unittest.py";
    nerr += build_and_write_matrix_zxy(
      opts, halfspace ? &A_zxy_half : &A_zxy_full);
  }
  if (A_zxy_full.size() != A_zxy_half.size() || A_zxy_full.empty()) {
    printf("matrix_zxy halfspace test: incompatible matrix sizes\n");
    ++nerr;
  } else {
    Real max_diff = 0;
    for (size_t i = 0; i < A_zxy_full.size(); ++i) {
      if (not std::isfinite(A_zxy_full[i]) ||
          not std::isfinite(A_zxy_half[i])) {
        printf("matrix_zxy halfspace test: nonfinite matrix entry\n");
        ++nerr;
        break;
      }
      max_diff = std::max(
        max_diff, std::abs(A_zxy_full[i] - A_zxy_half[i]));
    }
    if (max_diff <= 1e3*mv3::eps) {
      printf("matrix_zxy halfspace test: halfspace had no measurable effect\n");
      ++nerr;
    }
  }
  return nerr;
}

static int test_matrix_form_zxy_quads () {
  int nerr = 0;
  for (const std::string surface : {std::string("yxz"), std::string("zxy")})
    for (const bool halfspace : {false, true}) {
      MatrixZxyOpts opts;
      opts.surface = surface;
      opts.element = "quad";
      opts.nx = 2;
      opts.ny = 2;
      opts.nz = 2;
      opts.lx = 2;
      opts.ly = 3;
      opts.lz = 1;
      opts.testcase = 20;
      opts.disloc_comp = 0;
      opts.halfspace = halfspace;
      opts.verify = true;
      opts.outfile = "/tmp/ctzx_" + surface + "_quad_matrix_" +
        (halfspace ? "halfspace" : "fullspace") + "_unittest.py";
      RealArray A;
      nerr += build_and_write_matrix_zxy(opts, &A);
      const int nc = opts.nx*(surface == "yxz" ? opts.nz : opts.ny);
      if (A.size() != size_t(matrix_zxy_ncomp*nc*nc)) {
        printf("matrix_%s quad test: wrong matrix size\n", surface.c_str());
        ++nerr;
      }
    }
  return nerr;
}

static int test_matrix_form_zxy_bicubic_recon () {
  int nerr = 0;
  for (const bool halfspace : {false, true}) {
    MatrixZxyOpts opts;
    opts.surface = "zxy";
    opts.element = "quad";
    opts.srfrecon = true;
    opts.nx = 2;
    opts.ny = 2;
    opts.lx = 2;
    opts.ly = 3;
    opts.testcase = 70;
    opts.disloc_comp = 0;
    opts.halfspace = halfspace;
    opts.verify = true;
    opts.outfile = std::string("/tmp/ctzx_zxy_bicubic_matrix_") +
      (halfspace ? "halfspace" : "fullspace") + "_unittest.py";
    RealArray A;
    nerr += build_and_write_matrix_zxy(opts, &A);
    const int nc = opts.nx*opts.ny;
    if (A.size() != size_t(matrix_zxy_ncomp*nc*nc)) {
      printf("matrix_zxy bicubic test: wrong matrix size\n");
      ++nerr;
    }
    for (const auto v : A)
      if (!std::isfinite(v)) {
        printf("matrix_zxy bicubic test: nonfinite matrix entry\n");
        ++nerr;
        break;
      }
  }
  return nerr;
}

static int test_matrix_form_zxy_simple_output () {
  int nerr = 0;
  MatrixZxyOpts opts;
  opts.surface = "zxy";
  opts.element = "quad";
  opts.nx = 1;
  opts.ny = 1;
  opts.testcase = 70;
  opts.simple_output = true;
  opts.outfile = "/tmp/ctzx_zxy_simple_output_unittest.py";
  nerr += build_and_write_matrix_zxy(opts);

  std::ifstream is(opts.outfile);
  const std::string text((std::istreambuf_iterator<char>(is)),
                         std::istreambuf_iterator<char>());
  const auto require = [&] (const char* token) {
    if (text.find(token) == std::string::npos) {
      printf("matrix_zxy simple output: missing %s\n", token);
      ++nerr;
    }
  };
  const auto reject = [&] (const char* token) {
    if (text.find(token) != std::string::npos) {
      printf("matrix_zxy simple output: unexpected %s\n", token);
      ++nerr;
    }
  };
  require("d['m']");
  require("d['A_xz']");
  require("d['A_zz']");
  reject("d['testcase']");
  reject("d['disloc']");
  reject("d['sigma']");
  reject("d['bezier_ctrl_z']");
  return nerr;
}

static int test_hmatrix () {
  int nerr = 0;

#ifdef WOODLAND_HAVE_HMMVP
  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(0, zshape, disloc);

  ConvTest ct;
  ct.init(zshape, disloc);
  ct.set_verbosity(0);
  ct.set_use_surface_recon(0);
  ct.set_nx(4);
  ct.set_use_halfspace(false);
  ct.set_use_nonunirect(true);
  ct.discretize();

  RealArray dd, sd;
  ct.eval(dd, sd, ConvTest::EvalMethod::direct);

  const auto d = ct.get_discretization();
  RealArray sh;
  nerr += hmat::Hmatrices::unittest(d, &ct.get_gfp(), &dd, &sh);

  nerr += check_errors(sd, sh, 1e5*mv3::eps, 1e-6);
#endif

  return nerr;
}

// Yudong 02/20/2026
// Test triangulation of zxy surfaces. Writes mesh to ./triangulate_zxy_mesh.py
// for plotting: python ./triangulate_zxy_mesh.py
static int test_triangulate_zxy () {
  int nerr = 0;
  const int nx = 20, ny = 20;
  const auto zxy_shape = gallery::ZxyFn::Shape::trig1_cosy;

  auto t = ConvTest::triangulate_zxy(nx, ny, zxy_shape, 2);

  const int pnx = nx + 1;
  const int ncorner = (nx + 1) * (ny + 1);
  const int ntri = nx * ny * 2;
  if (t->get_nvtx() != ncorner) {
    printf("triangulate_zxy: nvtx %d expected %d\n",
           int(t->get_nvtx()), ncorner);
    ++nerr;
  }
  if (t->get_ncell() != ntri) {
    printf("triangulate_zxy: ncell %d expected %d\n",
           int(t->get_ncell()), ntri);
    ++nerr;
  }

  Real z00, z11, z_mid;
  gallery::ZxyFn::eval(zxy_shape, 0, 0, z00, nullptr, nullptr);
  gallery::ZxyFn::eval(zxy_shape, 1, 1, z11, nullptr, nullptr);
  gallery::ZxyFn::eval(zxy_shape, 0.5, 0.5, z_mid, nullptr, nullptr);

  const Real* v00 = t->get_vtx(0);
  const Real* v11 = t->get_vtx(pnx * ny + nx);
  const Real* v_mid = t->get_vtx(pnx * (ny/2) + nx/2);

  const Real tol = 1e-14;
  if (std::abs(v00[0]) > tol || std::abs(v00[1]) > tol ||
      std::abs(v00[2] - z00) > tol) {
    printf("triangulate_zxy: vtx(0,0) got (%.2e,%.2e,%.2e) expected (0,0,%.2e)\n",
           v00[0], v00[1], v00[2], z00);
    ++nerr;
  }
  if (std::abs(v11[0] - 1) > tol || std::abs(v11[1] - 1) > tol ||
      std::abs(v11[2] - z11) > tol) {
    printf("triangulate_zxy: vtx(1,1) got (%.2e,%.2e,%.2e) expected (1,1,%.2e)\n",
           v11[0], v11[1], v11[2], z11);
    ++nerr;
  }
  if (std::abs(v_mid[0] - 0.5) > tol || std::abs(v_mid[1] - 0.5) > tol ||
      std::abs(v_mid[2] - z_mid) > tol) {
    printf("triangulate_zxy: vtx(0.5,0.5) got (%.2e,%.2e,%.2e) expected (0.5,0.5,%.2e)\n",
           v_mid[0], v_mid[1], v_mid[2], z_mid);
    ++nerr;
  }

  auto t4 = ConvTest::triangulate_zxy(nx, ny, zxy_shape, 4);
  const int ncorner4 = (nx+1)*(ny+1);
  const int ncenter4 = nx*ny;
  if (t4->get_nvtx() != ncorner4 + ncenter4) {
    printf("triangulate_zxy ntri=4: nvtx %d expected %d\n",
           int(t4->get_nvtx()), ncorner4 + ncenter4);
    ++nerr;
  }
  if (t4->get_ncell() != nx*ny*4) {
    printf("triangulate_zxy ntri=4: ncell %d expected %d\n",
           int(t4->get_ncell()), nx*ny*4);
    ++nerr;
  }

  // Write triangulation to Python file for plotting (./ created by unittest).
  {
    FILE* fp = fopen("./triangulate_zxy_mesh.py", "w");
    if (fp) {
      pywrite_header(fp);
      fprintf(fp, "# Triangulation from triangulate_zxy(nx=%d, ny=%d, trig1_cosy, ntri_per_rect=2)\n", nx, ny);
      fprintf(fp, "m = {}\n");
      pywrite(fp, "m", *t);
      fprintf(fp, "\nif __name__ == \"__main__\":\n");
      fprintf(fp, "  import matplotlib.pyplot as plt\n");
      fprintf(fp, "  from mpl_toolkits.mplot3d import Axes3D\n");
      fprintf(fp, "  vtx, topo = m['vtx'], m['topo']\n");
      fprintf(fp, "  fig = plt.figure()\n");
      fprintf(fp, "  ax = fig.add_subplot(111, projection='3d')\n");
      fprintf(fp, "  ax.plot_trisurf(vtx[:,0], vtx[:,1], vtx[:,2], triangles=topo, edgecolor='w', linewidth=0.1)\n");
      fprintf(fp, "  ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')\n");
      fprintf(fp, "  plt.savefig('./triangulate_zxy_mesh.png')\n");
      fprintf(fp, "  plt.show()\n");
      fclose(fp);
    }
  }

  return nerr;
}

static int test_req0_and_scale () {
  int nerr = 0;

  ZxFn::Shape zshape;
  Disloc::Ptr disloc;
  setup(0, zshape, disloc);

  ConvTest ct;
  ct.init(zshape, disloc);
  ct.set_verbosity(0);
  ct.set_use_surface_recon(0);
  ct.set_use_exact_tangents(1);
  ct.set_disloc_order(2);
  ct.set_use_nonunirect(false);

  for (const bool use_halfspace : {false, true}) {
    ct.set_use_halfspace(use_halfspace);
    ct.set_nx(use_halfspace ? 5 : 6);
    ct.set_ny(3);
    RealArray dr, sr; // ref
    ct.set_mesh_scale(false);
    ct.set_support_req0(false);
    ct.eval(dr, sr, ConvTest::EvalMethod::direct);
    for (const bool scale : {false, true})
      for (const bool req0 : {false, true}) {
        if (not (scale or req0)) continue; // same as ref
        ct.set_mesh_scale(scale);
        ct.set_support_req0(req0);
        RealArray dt, st;
        ct.eval(dt, st, ConvTest::EvalMethod::direct);
        const int ne =
          scale ?
          check_errors(sr, st, 5e-5, 3e-3, 10) :
          check_errors(sr, st, 1e-8, 5e-7, 10);
        if (ne)
          printf("use_halfspace %d scale %d req0 %d\n",
                 int(use_halfspace), int(scale), int(req0));
        nerr += ne;
      }
  }

  return nerr;
}

int ConvTest::unittest () {
  int nerr = 0, ne;
  rununittest(ZxFn::unittest);
  rununittest(test_triangulate_zxy);
  rununittest(ExtrudedCubicSplineSurface::unittest);
  rununittest(FlatElementSurface::unittest);
  rununittest(ConvTest::test_interp);
  rununittest(test_req0_and_scale);
#ifdef WOODLAND_ACORN_HAVE_DC3D
  nerr += test_fast_okada_methods(false);
#endif
  nerr += test_fast_okada_methods(true);
  rununittest(test_fast_woodland_methods);
  rununittest(test_fast_nonunirect);
  rununittest(test_matrix_form);
  rununittest(test_yxz_geometry);
  rununittest(test_zxy_scaled_geometry);
  rununittest(test_quad_mesh_zxy_yxz);
  rununittest(test_matrix_form_zxy);
  rununittest(test_matrix_form_zxy_quads);
  rununittest(test_matrix_form_zxy_bicubic_recon);
  rununittest(test_matrix_form_zxy_simple_output);
  rununittest(test_hmatrix);
  return nerr;
}

} // namespace ctzx
} // namespace squirrel
} // namespace woodland
