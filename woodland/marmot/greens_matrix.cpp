#include "woodland/marmot/greens_matrix.hpp"

#include "woodland/acorn/dbg.hpp"
#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/squirrel/ctzx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace woodland {
namespace marmot {

typedef acorn::Matvec<3,Real> mv3;
using squirrel::Discretization;
using squirrel::Stress;

static const char* stress_comp_names[6] = {"xx", "xy", "xz", "yy", "yz", "zz"};

MatrixOptions MatrixOptions::parse (const std::string& params) {
  MatrixOptions o;
  for (const auto& tok : acorn::split(params, ",")) {
    if (tok.empty()) continue;
    const auto kv = acorn::split(tok, "=");
    throw_if(kv.size() != 2, "marmot: expected key=value, got " + tok);
    const auto& key = kv[0];
    const auto& val = kv[1];
    try {
      if (key == "geom") o.geom = val;
      else if (key == "out") o.out = val;
      else if (key == "lcs") o.lcs = val;
      else if (key == "nx") o.nx = std::stoi(val);
      else if (key == "ny") o.ny = std::stoi(val);
      else if (key == "lam") o.lam = std::stod(val);
      else if (key == "mu") o.mu = std::stod(val);
      else if (key == "halfspace") o.halfspace = std::stoi(val) != 0;
      else if (key == "dislocorder") o.dislocorder = std::stoi(val);
      else if (key == "disloc") {
        o.disloc_comps.clear();
        for (const char c : val) {
          throw_if(c < '0' || c > '2' ||
                   std::count(o.disloc_comps.begin(), o.disloc_comps.end(),
                              c - '0'),
                   "marmot: disloc must be distinct digits from 0, 1, 2");
          o.disloc_comps.push_back(c - '0');
        }
      } else if (key == "slip") {
        const auto c = acorn::split(val, ":");
        throw_if(c.size() != 3, "marmot: slip=a:b:c");
        for (int k = 0; k < 3; ++k) o.slip[k] = std::stod(c[k]);
      } else if (key == "comps") {
        if (val == "traction") o.stress_comps = {2, 4, 5};
        else if (val == "all") o.stress_comps = {0, 1, 2, 3, 4, 5};
        else throw_if(true, "marmot: comps must be traction or all");
      } else throw_if(true, "marmot: unrecognized option " + key);
    } catch (const std::invalid_argument&) {
      throw_if(true, "marmot: invalid value in " + tok);
    }
  }
  throw_if(o.disloc_comps.empty(), "marmot: disloc is empty");
  throw_if(o.dislocorder < 0 || o.dislocorder > 3,
           "marmot: dislocorder must be 0, 1, 2, or 3");
  throw_if(not (o.lam >= 0 && o.mu > 0), "marmot: need lam >= 0, mu > 0");
  throw_if(o.nx == 0 || o.ny == 0 || o.nx < -1 || o.ny < -1,
           "marmot: nx and ny must be positive");
  BezierPatchSurface::convert(o.lcs); // validate
  return o;
}

struct MatrixEvaluator::Impl {
  MatrixOptions o;
  Geometry::CPtr g;
  BezierPatchSurface::Ptr srf;
  Discretization::Ptr d;
  Stress::Ptr stress;
  Stress::Options so, so_nbr;
  squirrel::GreensFnParams gfp;
};

MatrixEvaluator::MatrixEvaluator (const MatrixOptions& o) {
  throw_if(o.geom.empty(), "marmot: geom=FILE is required");
  init(GridGeometry::read(o.geom), o);
}

MatrixEvaluator::MatrixEvaluator (const Geometry::CPtr& g,
                                  const MatrixOptions& o) {
  init(g, o);
}

void MatrixEvaluator::init (const Geometry::CPtr& g, const MatrixOptions& o) {
  p = std::make_shared<Impl>();
  p->o = o;
  p->g = g;
  Real sd[2], td[2];
  g->get_domain(sd, td);
  const auto gg = dynamic_cast<const GridGeometry*>(g.get());
  throw_if((o.nx < 0 || o.ny < 0) && not gg,
           "marmot: nx and ny are required for this geometry");
  const auto sb = o.nx > 0 ? uniform_breaks(sd[0], sd[1], o.nx) : gg->get_s();
  const auto tb = o.ny > 0 ? uniform_breaks(td[0], td[1], o.ny) : gg->get_t();
  p->srf = std::make_shared<BezierPatchSurface>(
    *g, sb, tb, BezierPatchSurface::convert(o.lcs));
  const auto& m = p->srf->get_mesh();
  if (o.halfspace) {
    // The surface, not just the nodes, must lie in z <= 0. Bezier control
    // points bound the patch (convex hull property).
    const auto& c = p->srf->get_control_points();
    Real zmax = -1e300, scale = 0;
    for (size_t k = 0; k < c.size(); k += 48)
      for (int q = 0; q < 16; ++q) {
        zmax = std::max(zmax, c[k + 32 + q]);
        for (int d = 0; d < 3; ++d)
          scale = std::max(scale, std::abs(c[k + 16*d + q]));
      }
    throw_if(zmax > 1e-10*scale,
             "marmot: halfspace=1 requires the fault to lie in z <= 0; max z "
             "of the Bezier control points is " + std::to_string(zmax));
  }
  p->d = squirrel::make_discretization(m, p->srf);
  p->d->set_disloc_order(o.dislocorder);
  p->stress = Stress::create(p->d);
  squirrel::ctzx::set_standard_options(p->so, p->so_nbr);
  p->gfp = squirrel::GreensFnParams(o.lam, o.mu, o.halfspace);
}

Idx MatrixEvaluator::get_ncell () const { return p->srf->get_ncell(); }
const MatrixOptions& MatrixEvaluator::get_options () const { return p->o; }
const Geometry& MatrixEvaluator::get_geometry () const { return *p->g; }
const BezierPatchSurface& MatrixEvaluator::get_surface () const {
  return *p->srf;
}
Discretization::CPtr MatrixEvaluator::get_discretization () const {
  return p->d;
}
const squirrel::GreensFnParams& MatrixEvaluator::get_gfp () const {
  return p->gfp;
}

void MatrixEvaluator
::calc_entries (const Idx isrc, const Idx ircv, const int disloc_comp,
                Real sigma[6]) const {
  const Idx nc = get_ncell();
  throw_if(isrc < 0 || isrc >= nc || ircv < 0 || ircv >= nc,
           "MatrixEvaluator::calc_entries: cell index out of range");
  throw_if(disloc_comp < 0 || disloc_comp > 2,
           "MatrixEvaluator::calc_entries: disloc_comp must be 0, 1, or 2");
  p->stress->calc_matrix_entries(p->gfp, isrc, ircv, disloc_comp, sigma,
                                 p->so, p->so_nbr);
}

void build_matrix (const MatrixEvaluator& e, RealArray& G) {
  const auto& o = e.get_options();
  const Idx nc = e.get_ncell();
  const int nd = int(o.disloc_comps.size()), ns = int(o.stress_comps.size());
  G.resize(size_t(nd)*ns*nc*nc);
  ompparfor for (Idx isrc = 0; isrc < nc; ++isrc)
    for (Idx ircv = 0; ircv < nc; ++ircv)
      for (int di = 0; di < nd; ++di) {
        Real s[6];
        e.calc_entries(isrc, ircv, o.disloc_comps[di], s);
        for (int ci = 0; ci < ns; ++ci)
          G[((size_t(di)*ns + ci)*nc + ircv)*nc + isrc] = s[o.stress_comps[ci]];
      }
}

static void write_array (FILE* fp, const std::string& var, const size_t m,
                         const int n, CRPtr a) {
  fprintf(fp, "%s = npy.array([\n", var.c_str());
  for (size_t i = 0; i < m; ++i) {
    fprintf(fp, "[");
    for (int j = 0; j < n; ++j) fprintf(fp, "%.17g,", a[i*n + j]);
    fprintf(fp, "],\n");
  }
  fprintf(fp, "])\n");
}

static std::string basename (const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos+1);
}

void write_output (const MatrixEvaluator& e, const RealArray* G) {
  const auto& o = e.get_options();
  const auto& srf = e.get_surface();
  const auto& m = *srf.get_mesh();
  const Idx nc = e.get_ncell();
  const int nd = int(o.disloc_comps.size()), ns = int(o.stress_comps.size());
  const std::string pyname = o.out + ".py", binname = o.out + "_G.bin";

  if (G) {
    throw_if(G->size() != size_t(nd)*ns*nc*nc, "write_output: G size");
    FILE* fp = fopen(binname.c_str(), "wb");
    throw_if(not fp, "marmot: cannot open " + binname);
    const size_t nw = fwrite(G->data(), sizeof(Real), G->size(), fp);
    fclose(fp);
    throw_if(nw != G->size(), "marmot: short write to " + binname);
  }

  FILE* fp = fopen(pyname.c_str(), "w");
  throw_if(not fp, "marmot: cannot open " + pyname);
  fprintf(fp, "import os\nimport numpy as npy\n");
  fprintf(fp, "d = {}\n");
  fprintf(fp, "d['geometry'] = '%s'\n", e.get_geometry().name());
  fprintf(fp, "d['geom_file'] = %s\n",
          o.geom.empty() ? "None" : ("'" + o.geom + "'").c_str());
  fprintf(fp, "d['ncell'] = %d\n", int(nc));
  fprintf(fp, "d['nx'] = %d\n", srf.get_nx());
  fprintf(fp, "d['ny'] = %d\n", srf.get_ny());
  fprintf(fp, "d['cell_index'] = 'j*nx + i, i along s, j along t'\n");
  fprintf(fp, "d['lam'] = %.17g\n", o.lam);
  fprintf(fp, "d['mu'] = %.17g\n", o.mu);
  fprintf(fp, "d['halfspace'] = %s\n", o.halfspace ? "True" : "False");
  fprintf(fp, "d['dislocorder'] = %d\n", o.dislocorder);
  fprintf(fp, "d['lcs_primary'] = '%s'\n", o.lcs.c_str());
  fprintf(fp, "d['disloc_comps'] = [");
  for (const int c : o.disloc_comps) fprintf(fp, "%d, ", c);
  fprintf(fp, "]\n");
  fprintf(fp, "d['stress_comps'] = [");
  for (const int c : o.stress_comps) fprintf(fp, "'%s', ", stress_comp_names[c]);
  fprintf(fp, "]\n");
  write_array(fp, "d['sb']", 1, srf.get_nx()+1, srf.get_sb().data());
  fprintf(fp, "d['sb'] = d['sb'][0]\n");
  write_array(fp, "d['tb']", 1, srf.get_ny()+1, srf.get_tb().data());
  fprintf(fp, "d['tb'] = d['tb'][0]\n");

  fprintf(fp, "d['m'] = {}\n");
  {
    const auto& to = *m.get_topo();
    fprintf(fp, "d['m']['topo'] = npy.array([\n");
    for (Idx ci = 0; ci < nc; ++ci) {
      const auto c = to.get_cell(ci);
      fprintf(fp, "[%d,%d,%d,%d],\n", int(c[0]), int(c[1]), int(c[2]),
              int(c[3]));
    }
    fprintf(fp, "], dtype=npy.int32)\n");
    const auto& v = *m.get_vtxs();
    write_array(fp, "d['m']['vtx']", v.get_nvtx(), 3, v.get_vtxs());
  }
  RealArray ctr(3*nc), lcs(9*nc), area(nc);
  acorn::Workspace w;
  for (Idx ci = 0; ci < nc; ++ci) {
    srf.cell_ctr_xyz(ci, &ctr[3*ci]);
    srf.cell_ctr_lcs(ci, &lcs[9*ci]);
    area[ci] = squirrel::cell_surface_area(w, srf, ci);
  }
  write_array(fp, "d['ctr']", nc, 3, ctr.data());
  write_array(fp, "d['lcs']", nc, 9, lcs.data());
  fprintf(fp, "d['lcs_layout'] = 'xhat, yhat, zhat (normal)'\n");
  write_array(fp, "d['area']", 1, nc, area.data());
  fprintf(fp, "d['area'] = d['area'][0]\n");
  write_array(fp, "d['bezier_ctrl']", nc, 48,
              srf.get_control_points().data());
  fprintf(fp, "d['bezier_ctrl_layout'] = 'cell x (x,y,z) x (4*j+i), "
          "i along s'\n");
  if (G) {
    fprintf(fp, "d['G_file'] = '%s'\n", basename(binname).c_str());
    fprintf(fp, "d['G_shape'] = (%d, %d, %d, %d)\n", nd, ns, int(nc), int(nc));
    fprintf(fp, "d['G_layout'] = 'disloc comp, stress comp, receiver, "
            "source; float64, native byte order'\n");
    fprintf(fp,
            "def load_G():\n"
            "    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),"
            " d['G_file'])\n"
            "    return npy.fromfile(fn, dtype=npy.float64)"
            ".reshape(d['G_shape'])\n");
  }
  fclose(fp);
}

void run_matrix (const std::string& params) {
  const auto o = MatrixOptions::parse(params);
  Real t0 = acorn::dbg::gettime();
  MatrixEvaluator e(o);
  const Idx nc = e.get_ncell();
  printf("marmot matrix: geom %s ncell %d (nx %d ny %d) lcs %s lam %g mu %g "
         "halfspace %d dislocorder %d disloc %d comps %s\n",
         o.geom.c_str(), int(nc), e.get_surface().get_nx(),
         e.get_surface().get_ny(), o.lcs.c_str(), o.lam, o.mu,
         int(o.halfspace), o.dislocorder, int(o.disloc_comps.size()),
         o.comps_name().c_str());
  const double nbytes = 8.0*o.disloc_comps.size()*o.stress_comps.size()*nc*nc;
  if (nbytes > 1e9) printf("marmot matrix: G needs %.2e bytes\n", nbytes);
  Real t1 = acorn::dbg::gettime();
  printf("setup et %1.3e\n", t1 - t0);
  RealArray G;
  build_matrix(e, G);
  Real t2 = acorn::dbg::gettime();
  printf("fill matrix et %1.3e\n", t2 - t1);
  write_output(e, &G);
  printf("wrote %s.py %s_G.bin\n", o.out.c_str(), o.out.c_str());
}

void calc_stress (const MatrixEvaluator& e, CRPtr dislocs, RealArray& sigma) {
  const auto d = e.get_discretization();
  const auto& m = *d->get_mesh();
  const auto& re = *m.get_relations();
  const auto& srf = *d->get_surface();
  const auto& gfp = e.get_gfp();
  const Idx nc = e.get_ncell();
  Stress stress(d);
  stress.init_dislocs(dislocs);
  Stress::Options so, so_nbr;
  squirrel::ctzx::set_standard_options(so, so_nbr);
  sigma.assign(6*size_t(nc), 0);
  ompparfor for (Idx ircv = 0; ircv < nc; ++ircv) {
    Real acc[6] = {0};
    for (Idx isrc = 0; isrc < nc; ++isrc) {
      Real s[6];
      stress.calc_s1_r1(gfp, isrc, ircv, s,
                        squirrel::mesh::are_nbrs(re, ircv, isrc) ? so_nbr : so);
      for (int k = 0; k < 6; ++k) acc[k] += s[k];
    }
    Real lcs[9];
    srf.cell_ctr_lcs(ircv, lcs);
    acorn::rotate_sym_tensor_3x3_RARt(lcs, acc);
    for (int k = 0; k < 6; ++k) sigma[6*size_t(ircv) + k] = acc[k];
  }
}

void run_stress (const std::string& params) {
  const auto o = MatrixOptions::parse(params);
  Real t0 = acorn::dbg::gettime();
  MatrixEvaluator e(o);
  const Idx nc = e.get_ncell();
  printf("marmot stress: geom %s ncell %d (nx %d ny %d) lcs %s lam %g mu %g "
         "halfspace %d dislocorder %d slip %g:%g:%g\n", o.geom.c_str(),
         int(nc), e.get_surface().get_nx(), e.get_surface().get_ny(),
         o.lcs.c_str(), o.lam, o.mu, int(o.halfspace), o.dislocorder,
         o.slip[0], o.slip[1], o.slip[2]);
  RealArray dislocs(3*size_t(nc)), sigma;
  for (Idx ci = 0; ci < nc; ++ci)
    for (int k = 0; k < 3; ++k) dislocs[3*ci + k] = o.slip[k];
  Real t1 = acorn::dbg::gettime();
  calc_stress(e, dislocs.data(), sigma);
  Real t2 = acorn::dbg::gettime();
  printf("setup et %1.3e stress et %1.3e\n", t1 - t0, t2 - t1);
  write_output(e, nullptr);
  FILE* fp = fopen((o.out + ".py").c_str(), "a");
  throw_if(not fp, "marmot: cannot open " + o.out + ".py");
  fprintf(fp, "d['slip'] = [%.17g, %.17g, %.17g]\n", o.slip[0], o.slip[1],
          o.slip[2]);
  write_array(fp, "d['sigma']", nc, 6, sigma.data());
  fprintf(fp, "d['sigma_comps'] = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']\n");
  fprintf(fp, "d['sigma_layout'] = 'cell x comp, receiver LCS; "
          "normal stress is zz'\n");
  fclose(fp);
  Real smin = 1e300, smax = -1e300;
  for (Idx ci = 0; ci < nc; ++ci) {
    smin = std::min(smin, sigma[6*ci + 5]);
    smax = std::max(smax, sigma[6*ci + 5]);
  }
  printf("sigma_zz (normal) min %1.4e max %1.4e\n", smin, smax);
  printf("wrote %s.py\n", o.out.c_str());
}

void run_mesh (const std::string& params) {
  const auto o = MatrixOptions::parse(params);
  MatrixEvaluator e(o);
  write_output(e, nullptr);
  printf("wrote %s.py (ncell %d)\n", o.out.c_str(), int(e.get_ncell()));
}

// A curved listric fault: the strike trace bends in map view, and the dip
// decreases from 70 to 35 degrees down dip. i is along strike, j down dip, so
// X_s x X_t points up into the hanging wall.
static void make_example_geom (const int ni, const int nj, RealArray& xyz) {
  const Real L = 2, W = 1, top = 0.1, bend = 0.2;
  const Real dip0 = 70*M_PI/180, dip1 = 35*M_PI/180, k = (dip0 - dip1)/W;
  xyz.resize(3*ni*nj);
  for (int j = 0; j < nj; ++j) {
    const Real w = W*j/(nj-1);
    const Real h = (std::sin(dip0) - std::sin(dip0 - k*w))/k;
    const Real dz = (std::cos(dip0 - k*w) - std::cos(dip0))/k;
    for (int i = 0; i < ni; ++i) {
      const Real a = L*i/(ni-1);
      const Real y0 = bend*std::sin(M_PI*a/L);
      const Real yp = bend*M_PI/L*std::cos(M_PI*a/L);
      const Real nn = std::sqrt(1 + yp*yp);
      auto p = &xyz[3*(j*ni + i)];
      p[0] = a - h*yp/nn;
      p[1] = y0 + h/nn;
      p[2] = -top - dz;
    }
  }
}

void run_example_geom (const std::string& params) {
  int ni = 33, nj = 17;
  std::string out = "marmot_example_geom.txt";
  for (const auto& tok : acorn::split(params, ",")) {
    if (tok.empty()) continue;
    const auto kv = acorn::split(tok, "=");
    throw_if(kv.size() != 2, "example_geom: expected key=value");
    if (kv[0] == "out") out = kv[1];
    else if (kv[0] == "n") {
      const auto n = acorn::split(kv[1], ":");
      throw_if(n.size() != 2, "example_geom: n=ni:nj");
      ni = std::stoi(n[0]); nj = std::stoi(n[1]);
    } else throw_if(true, "example_geom: unrecognized option " + kv[0]);
  }
  throw_if(ni < 2 || nj < 2, "example_geom: need ni, nj >= 2");
  RealArray xyz;
  make_example_geom(ni, nj, xyz);
  GridGeometry::write(out, ni, nj, xyz.data());
  printf("wrote %s (%d x %d points)\n", out.c_str(), ni, nj);
}

// ---------------------------------------------------------------------------
// Unit tests.

static Real max_abs (const RealArray& a) {
  Real m = 0;
  for (const auto e : a) m = std::max(m, std::abs(e));
  return m;
}

// On z = f(x,y) with lcs=x, marmot's patches and LCSs are those of squirrel's
// BicubicBezierSurface, so the matrices must agree to rounding. The h.f.p.
// self terms amplify rounding differences (marmot evaluates x and y from
// Bezier control points; squirrel uses them exactly) to ~1e-10 relative.
static int test_vs_squirrel () {
  using squirrel::gallery::ZxyFn;
  int nerr = 0;
  struct Case { ZxyFn::Shape shape; int testcase; Real amp; bool hs; };
  const Case cases[] = {{ZxyFn::Shape::cos10_cos12_y, 70, 0.05, false},
                        {ZxyFn::Shape::trig1_cosy, 20, 0, true}};
  const int nx = 3, ny = 2;
  const Real lx = 2, ly = 1.5;
  for (const auto& c : cases) {
    // Rounding noise is relative to the largest entries of the case, so
    // measure the error of all dislocation components against that scale.
    Real err[3], scale = 0;
    for (int dc = 0; dc < 3; ++dc) {
      char buf[512];
      snprintf(buf, sizeof(buf), "surface=zxy,element=quad,srfrecon=1,"
               "nx=%d,ny=%d,lx=%g,ly=%g,testcase=%d,amp=%g,disloccomp=%d,"
               "halfspace=%d", nx, ny, lx, ly, c.testcase, c.amp, dc,
               int(c.hs));
      // Sets the process-wide cos10_cos12 amplitude, used by ZxyGeometry.
      squirrel::ctzx::MatrixZxyEvaluator se(buf);
      MatrixOptions o;
      o.nx = nx; o.ny = ny; o.lcs = "x"; o.halfspace = c.hs;
      o.disloc_comps = {dc};
      o.stress_comps = {2, 5};
      MatrixEvaluator me(std::make_shared<ZxyGeometry>(c.shape, lx, ly), o);
      RealArray G;
      build_matrix(me, G);
      const Idx nc = me.get_ncell();
      RealArray Gs(2*nc*nc);
      for (Idx si = 0; si < nc; ++si)
        for (Idx ri = 0; ri < nc; ++ri) {
          Real a[2];
          se.calc_entries(si, ri, a);
          for (int k = 0; k < 2; ++k) Gs[(k*nc + ri)*nc + si] = a[k];
        }
      err[dc] = 0;
      for (size_t k = 0; k < G.size(); ++k)
        err[dc] = std::max(err[dc], std::abs(G[k] - Gs[k]));
      scale = std::max(scale, max_abs(Gs));
    }
    for (int dc = 0; dc < 3; ++dc)
      if (not (err[dc] <= 1e-9*scale)) {
        printf("marmot vs squirrel: testcase %d disloc %d hs %d err %1.3e "
               "scale %1.3e\n", c.testcase, dc, int(c.hs), err[dc], scale);
        ++nerr;
      }
  }
  squirrel::gallery::ZxyFn::set_cos10_cos12_amp(0);
  return nerr;
}

// Rotating and translating a general (non-graph after rotation) fault must not
// change the whole-space matrices when the LCS follows the surface (lcs=s), up
// to rounding amplified by the h.f.p. self terms.
static int test_rotation_invariance () {
  int nerr = 0;
  const int ni = 5, nj = 4;
  RealArray xyz(3*ni*nj), xyz_rot(xyz.size());
  for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i) {
      auto p = &xyz[3*(j*ni + i)];
      const Real u = Real(i)/(ni-1), v = Real(j)/(nj-1);
      p[0] = 1.2*u;
      p[1] = 0.8*v + 0.1*u*u;
      p[2] = 0.15*std::sin(M_PI*u)*std::cos(0.5*M_PI*v) - 0.5;
    }
  // Rotation by 50 degrees about axis (1,2,-1)/sqrt(6), then a shift.
  Real ax[] = {1, 2, -1};
  mv3::normalize(ax);
  const Real th = 50*M_PI/180, ct = std::cos(th), st = std::sin(th);
  Real R[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      R[3*r+c] = (r == c ? ct : 0) + (1 - ct)*ax[r]*ax[c];
    }
  R[1] -= st*ax[2]; R[2] += st*ax[1];
  R[3] += st*ax[2]; R[5] -= st*ax[0];
  R[6] -= st*ax[1]; R[7] += st*ax[0];
  const Real shift[] = {3, -1, 2};
  for (int k = 0; k < ni*nj; ++k) {
    mv3::matvec(R, &xyz[3*k], &xyz_rot[3*k]);
    for (int d = 0; d < 3; ++d) xyz_rot[3*k+d] += shift[d];
  }
  MatrixOptions o;
  o.lcs = "s";
  o.stress_comps = {0, 1, 2, 3, 4, 5};
  RealArray G[2];
  for (int r = 0; r < 2; ++r) {
    const auto g = std::make_shared<GridGeometry>(
      ni, nj, r == 0 ? xyz.data() : xyz_rot.data());
    MatrixEvaluator e(g, o);
    build_matrix(e, G[r]);
  }
  Real err = 0;
  for (size_t k = 0; k < G[0].size(); ++k)
    err = std::max(err, std::abs(G[0][k] - G[1][k]));
  if (not (err <= 1e-8*max_abs(G[0]))) {
    printf("marmot rotation invariance: err %1.3e max %1.3e\n",
           err, max_abs(G[0]));
    ++nerr;
  }
  return nerr;
}

// The example listric fault in a half space: finite, nontrivial, the half
// space matters, and the output files are written.
static int test_example_and_output () {
  int nerr = 0;
  RealArray xyz;
  const int ni = 5, nj = 3;
  make_example_geom(ni, nj, xyz);
  const std::string gfn = "tmp/marmot_example_geom.txt";
  GridGeometry::write(gfn, ni, nj, xyz.data());
  RealArray G[2];
  for (int hs = 0; hs < 2; ++hs) {
    auto o = MatrixOptions::parse(
      "geom=" + gfn + ",out=tmp/marmot_unittest,disloc=02,halfspace="
      + std::to_string(hs));
    MatrixEvaluator e(o);
    build_matrix(e, G[hs]);
    for (const auto g : G[hs])
      if (not std::isfinite(g)) { ++nerr; break; }
    if (hs) {
      write_output(e, &G[hs]);
      FILE* fp = fopen("tmp/marmot_unittest_G.bin", "rb");
      if (not fp) ++nerr;
      else {
        fseek(fp, 0, SEEK_END);
        if (size_t(ftell(fp)) != 8*G[hs].size()) ++nerr;
        fclose(fp);
      }
    }
    // Self tractions are the largest and have the expected sign: shear
    // opposing the slip, normal opposing the opening.
    const Idx nc = e.get_ncell();
    for (Idx ci = 0; ci < nc; ++ci) {
      const Real txz = G[hs][((0*3 + 0)*nc + ci)*nc + ci];
      const Real tzz = G[hs][((1*3 + 2)*nc + ci)*nc + ci];
      if (not (txz < 0 && tzz < 0)) {
        printf("marmot example: cell %d self traction %1.3e %1.3e\n",
               int(ci), txz, tzz);
        ++nerr;
      }
    }
  }
  Real diff = 0;
  for (size_t k = 0; k < G[0].size(); ++k)
    diff = std::max(diff, std::abs(G[0][k] - G[1][k]));
  if (not (diff > 1e-3*max_abs(G[0]))) {
    printf("marmot example: halfspace had no effect\n");
    ++nerr;
  }
  return nerr;
}

// calc_stress for a uniform dislocation must equal G times the uniform vector:
// summing the matrix columns sums the reconstruction cardinal functions, which
// reproduce the constant.
static int test_stress_vs_matrix () {
  int nerr = 0;
  RealArray xyz;
  const int ni = 6, nj = 4;
  make_example_geom(ni, nj, xyz);
  for (int hs = 0; hs < 2; ++hs) {
    MatrixOptions o;
    o.halfspace = hs;
    o.disloc_comps = {0, 1, 2};
    o.stress_comps = {0, 1, 2, 3, 4, 5};
    MatrixEvaluator e(std::make_shared<GridGeometry>(ni, nj, xyz.data()), o);
    const Idx nc = e.get_ncell();
    RealArray G, sigma;
    build_matrix(e, G);
    const Real slip[3] = {0.7, -0.4, 0.2};
    RealArray dislocs(3*nc);
    for (Idx ci = 0; ci < nc; ++ci)
      for (int k = 0; k < 3; ++k) dislocs[3*ci + k] = slip[k];
    calc_stress(e, dislocs.data(), sigma);
    Real err = 0, scale = 0;
    for (Idx ri = 0; ri < nc; ++ri)
      for (int c = 0; c < 6; ++c) {
        Real g = 0;
        for (int di = 0; di < 3; ++di)
          for (Idx si = 0; si < nc; ++si)
            g += slip[di]*G[((size_t(di)*6 + c)*nc + ri)*nc + si];
        err = std::max(err, std::abs(g - sigma[6*ri + c]));
        scale = std::max(scale, std::abs(g));
      }
    if (not (err <= 1e-9*scale)) {
      printf("marmot calc_stress vs G: hs %d err %1.3e scale %1.3e\n",
             hs, err, scale);
      ++nerr;
    }
  }
  return nerr;
}

int unittest_matrix () {
  return test_stress_vs_matrix() + test_vs_squirrel() +
    test_rotation_invariance() +
    test_example_and_output();
}

} // namespace marmot
} // namespace woodland
