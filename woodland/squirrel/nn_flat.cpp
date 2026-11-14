// Nodal-normal test on flat triangles (EarthquakeCycle/nodalnormal/woodland_nn).
//
// Reads a triangulation, a normal mode and optionally vertex normals from a
// binary file, puts a dislocation of magnitude s (1 by default) along the
// local x direction of the slip frame on every triangle, and writes the stress
// at the centroids, resolved in the slip frame of the receiver.
//
// Input (little endian):
//   int32  nv, nt, mode, disloc_order, hill, has_slip
//   double A, lam, mu
//   double V[3 nv]; int32 T[3 nt]; double N[3 nv] if mode == 1,
//   double N6[18 nt] if mode == 3; double s[nt] if has_slip (the slip magnitude
//   at the centroids, else 1)
// mode 0: facet normal (the slip in the element plane, Woodland as it is);
//      1: barycentric blend of the vertex normals N;
//      2: exact normal of z = A f(x) g(y) at every point (hill 0: md,
//         1: c3; zero outside [0,1]^2);
//      3: quadratic blend of six normals per triangle N6 (vertices, then the
//         midpoints of the edges 01, 12, 20).
// Output: double [nt][9]: centroid (3), stress (xx, xy, xz, yy, yz, zz) in the
// slip frame of the receiver.

#include "woodland/squirrel/ctzx.hpp"
#include "woodland/squirrel/ctzx_surface.hpp"
#include "woodland/squirrel/ctzx_triangulation.hpp"
#include "woodland/squirrel/calc_stress.hpp"
#include "woodland/acorn/matvec.hpp"
#include "woodland/acorn/openmp.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/acorn/dbg.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace woodland {
namespace squirrel {
namespace ctzx {

namespace {
// f(x), g(y) of the hill and their first derivatives (nodalnormal/reference/
// smallslope.py: md, and c3 = md f sin^2(pi x), (1 - cos 2 pi y)^2).
void hill_fg (const int hill, const Real x, const Real y, Real& f, Real& fx,
              Real& g, Real& gy) {
  const Real a = 2*M_PI, b = 4*M_PI;
  const Real f0 = std::cos(a*x) - std::cos(b*x);
  const Real f1 = -a*std::sin(a*x) + b*std::sin(b*x);
  const Real g0 = 2 - 2*std::cos(a*y), g1 = 2*a*std::sin(a*y);
  if (hill == 0) {
    f = f0; fx = f1; g = g0; gy = g1;
    return;
  }
  const Real s = std::sin(M_PI*x), c = std::cos(M_PI*x);
  f = f0*s*s;
  fx = f1*s*s + f0*2*M_PI*s*c;
  const Real v0 = 0.5*g0, v1 = 0.5*g1;
  g = v0*v0;
  gy = 2*v0*v1;
}

template <typename T> void readn (FILE* fp, T* v, const size_t n) {
  if (std::fread(v, sizeof(T), n, fp) != n)
    throw std::runtime_error("nn_flat: short read");
}
} // namespace

void run_nn_flat (const std::string& params) {
  std::string fin, fout;
  for (const auto& tok : acorn::split(params, ",")) {
    const auto kv = acorn::split(tok, "=");
    if (kv.size() != 2) continue;
    if (kv[0] == "in") fin = kv[1];
    else if (kv[0] == "out") fout = kv[1];
  }
  throw_if(fin.empty() || fout.empty(), "nn_flat: need in=FILE,out=FILE");

  FILE* fp = std::fopen(fin.c_str(), "rb");
  throw_if(!fp, "nn_flat: cannot open the input file");
  int32_t hdr[6];
  readn(fp, hdr, 6);
  const int nv = hdr[0], nt = hdr[1], mode = hdr[2], dorder = hdr[3],
    hill = hdr[4], has_slip = hdr[5];
  double par[3];
  readn(fp, par, 3);
  const Real A = par[0], lam = par[1], mu = par[2];
  std::vector<double> V(3*nv), N;
  std::vector<int32_t> T(3*nt);
  readn(fp, V.data(), V.size());
  readn(fp, T.data(), T.size());
  if (mode == 1) {
    N.resize(3*nv);
    readn(fp, N.data(), N.size());
  } else if (mode == 3) {
    N.resize(18*nt);
    readn(fp, N.data(), N.size());
  }
  std::vector<double> slip(nt, 1.0);
  if (has_slip) readn(fp, slip.data(), slip.size());
  // has_slip == 2: also the exact slip magnitude at every quadrature point
  // (oracle; smallslope_var.slip = 1 + 0.5 cos(2 pi x) cos(2 pi y)).
  std::fclose(fp);

  const auto t = std::make_shared<Triangulation>();
  t->begin_modifications();
  for (int i = 0; i < nt; ++i) t->append_tri(&T[3*i]);
  for (int i = 0; i < nv; ++i) t->append_vtx();
  t->end_modifications();
  for (int i = 0; i < nv; ++i)
    for (int d = 0; d < 3; ++d) t->get_vtx(i)[d] = V[3*i+d];
  const auto tr = std::make_shared<TriangulationRelations>(t);
  const auto m = tri2mesh(*t, tr);

  const Real primary[] = {1, 0, 0};
  const auto srf = std::make_shared<FlatElementSurface>(t, primary);
  if (mode == 1)
    srf->set_vertex_normals(std::vector<Real>(N.begin(), N.end()));
  else if (mode == 3)
    srf->set_element_normals6(std::vector<Real>(N.begin(), N.end()));
  else if (mode == 2)
    srf->set_normal_fn([=] (const Real p[3], Real n[3]) {
      const Real x = p[0], y = p[1];
      Real hx = 0, hy = 0;
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        Real f, fx, g, gy;
        hill_fg(hill, x, y, f, fx, g, gy);
        hx = A*fx*g;
        hy = A*f*gy;
      }
      n[0] = -hx; n[1] = -hy; n[2] = 1;
    });
  if (has_slip == 2)
    srf->set_slip_fn([] (const Real p[3]) {
      return 1 + 0.5*std::cos(2*M_PI*p[0])*std::cos(2*M_PI*p[1]); });
  const auto d = make_discretization(m, srf);
  d->set_disloc_order(dorder);

  std::vector<Real> dislocs(3*nt, 0.0);
  for (int i = 0; i < nt; ++i) dislocs[3*i] = slip[i];

  Stress::Options o, onbr;
  set_standard_options(o, onbr);
  const GreensFnParams gfp(lam, mu, false);
  Stress stress(d);
  stress.init_dislocs(dislocs.data());
  const auto& re = *m->get_relations();

  std::printf("nn_flat: nv %d nt %d mode %d dislocorder %d hill %d A %g "
              "threads %d\n", nv, nt, mode, dorder, hill, A,
              acorn::get_max_threads());
  std::fflush(stdout);
  const Real t0 = acorn::dbg::gettime();
  std::vector<double> out(9*size_t(nt));
  int done = 0;
  ompparfor for (Idx ir = 0; ir < nt; ++ir) {
    Real sigma[6] = {0};
    for (Idx is = 0; is < nt; ++is) {
      Real s[6];
      const bool nbrs = mesh::are_nbrs(re, ir, is);
      stress.calc_s1_r1(gfp, is, ir, s, nbrs ? onbr : o);
      for (int i = 0; i < 6; ++i) sigma[i] += s[i];
    }
    Real lcs[9];
    srf->cell_ctr_slip_lcs(ir, lcs);
    acorn::rotate_sym_tensor_3x3_RARt(lcs, sigma);
    srf->cell_ctr_xyz(ir, &out[9*size_t(ir)]);
    for (int i = 0; i < 6; ++i) out[9*size_t(ir) + 3 + i] = sigma[i];
    int k;
#   pragma omp atomic capture
    k = ++done;
    if (nt >= 10 && k % (nt/10) == 0) {
      std::printf("nn_flat: %3d%% (%.1f s)\n", int(100.0*k/nt + 0.5),
                  acorn::dbg::gettime() - t0);
      std::fflush(stdout);
    }
  }
  fp = std::fopen(fout.c_str(), "wb");
  throw_if(!fp, "nn_flat: cannot open the output file");
  std::fwrite(out.data(), sizeof(double), out.size(), fp);
  std::fclose(fp);
  std::printf("nn_flat: done, %.1f s -> %s\n", acorn::dbg::gettime() - t0,
              fout.c_str());
}

} // namespace ctzx
} // namespace squirrel
} // namespace woodland
