#ifndef INCLUDE_WOODLAND_MARMOT_GREENS_MATRIX
#define INCLUDE_WOODLAND_MARMOT_GREENS_MATRIX

#include "woodland/marmot/bezier_surface.hpp"
#include "woodland/squirrel/calc_stress.hpp"

#include <string>

namespace woodland {
namespace marmot {

/* Options, given as a comma-separated list key=value:
     geom=FILE      fault geometry, a structured grid of 3D points; see
                    GridGeometry::read. Required by run_matrix and run_mesh.
     nx=, ny=       number of mesh cells along s (grid index i) and t (grid
                    index j), uniform in the chord-length parameters. Default:
                    the geometry grid itself is the mesh.
     lcs=           strike (default), x, or s: primary direction of the local
                    x axis; see BezierPatchSurface::LcsPrimary.
     lam=, mu=      Lame parameters [1, 1].
     halfspace=     0 (whole space, default) or 1 (half space z <= 0).
     dislocorder=   0, 1, 2 (default), 3: order of the dislocation
                    reconstruction from cell values.
     disloc=        dislocation components to compute, a string of digits
                    from 0 (local x), 1 (local y), 2 (opening). Default 012.
     comps=         traction (default): stress components xz, yz, zz, the
                    traction on the cell in its local frame; all: xx, xy, xz,
                    yy, yz, zz.
     out=PREFIX     output files PREFIX.py and PREFIX_G.bin [marmot_G].
     slip=a:b:c     stress command only: the dislocation, uniform over the
                    fault, in each cell's LCS components [1:0:0].
*/
struct MatrixOptions {
  std::string geom;
  int nx = -1, ny = -1;
  std::string lcs = "strike";
  Real lam = 1, mu = 1;
  bool halfspace = false;
  int dislocorder = 2;
  std::vector<int> disloc_comps = {0, 1, 2};
  std::vector<int> stress_comps = {2, 4, 5};
  std::string out = "marmot_G";
  Real slip[3] = {1, 0, 0};

  static MatrixOptions parse(const std::string& params);
  std::string comps_name () const {
    return stress_comps.size() == 6 ? "all" : "traction";
  }
};

// Green's function matrix entries on demand (e.g., for an H-matrix code).
//   Entry (ircv, isrc) for dislocation component d is the stress at the center
// of cell ircv, in ircv's LCS, due to the dislocation whose LCS component d has
// value 1 at the center of cell isrc and 0 at the centers of all other cells,
// with the dislocation over each cell reconstructed from the neighboring cell
// values (squirrel::Discretization). calc_entries is const and thread safe.
class MatrixEvaluator {
public:
  typedef std::shared_ptr<MatrixEvaluator> Ptr;

  // Geometry from o.geom.
  MatrixEvaluator(const MatrixOptions& o);
  // Given geometry; o.geom is ignored.
  MatrixEvaluator(const Geometry::CPtr& g, const MatrixOptions& o);

  Idx get_ncell() const;
  const MatrixOptions& get_options() const;
  const Geometry& get_geometry() const;
  const BezierPatchSurface& get_surface() const;
  squirrel::Discretization::CPtr get_discretization() const;
  const squirrel::GreensFnParams& get_gfp() const;

  // sigma = (xx, xy, xz, yy, yz, zz) in the receiver LCS.
  void calc_entries(const Idx isrc, const Idx ircv, const int disloc_comp,
                    Real sigma[6]) const;

private:
  struct Impl;
  std::shared_ptr<Impl> p;
  void init(const Geometry::CPtr& g, const MatrixOptions& o);
};

// G[((di*ns + ci)*nc + ircv)*nc + isrc] for di over o.disloc_comps and ci over
// o.stress_comps, with ns = o.stress_comps.size().
void build_matrix(const MatrixEvaluator& e, RealArray& G);

// Write PREFIX.py (mesh, cell centers, LCSs, areas, Bezier control points, and
// metadata) and, if G is nonnull, PREFIX_G.bin (G as float64 in native byte
// order, the layout of build_matrix).
void write_output(const MatrixEvaluator& e, const RealArray* G);

// Stress at every cell center, in the cell's LCS, sigma[6*ircv + (xx, xy, xz,
// yy, yz, zz)], due to the dislocation with LCS components dislocs[3*ci + k] at
// the cell centers, reconstructed as in the matrix. Equals G times dislocs but
// needs one integral per (source, receiver) pair instead of one per pair and
// source neighbor, so it is several times cheaper than forming G.
void calc_stress(const MatrixEvaluator& e, CRPtr dislocs, RealArray& sigma);

// Commands of woodland_marmot_unittest.
void run_matrix(const std::string& params);
void run_mesh(const std::string& params);
// Stress for the uniform dislocation slip= (no matrix): writes PREFIX.py with
// d['sigma'] (ncell x 6, receiver LCS) and the mesh metadata.
void run_stress(const std::string& params);
// Write an example curved fault geometry: params n=ni:nj [33:17], out=FILE.
void run_example_geom(const std::string& params);

int unittest_matrix();

} // namespace marmot
} // namespace woodland

#endif
