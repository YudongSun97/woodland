#include "woodland/marmot/unittest.hpp"
#include "woodland/marmot/geometry.hpp"
#include "woodland/marmot/bezier_surface.hpp"
#include "woodland/marmot/greens_matrix.hpp"

#include "woodland/acorn/dbg.hpp"
#include "woodland/acorn/macros.hpp"

#include <cstdio>
#include <sys/stat.h>

namespace woodland {
namespace marmot {

int unittest () {
  int nerr = 0, ne;
  // Some tests write files in tmp/.
  mkdir("tmp", 0777);
  rununittest(GridGeometry::unittest);
  rununittest(BezierPatchSurface::unittest);
  rununittest(unittest_matrix);
  printf("\n%s\n", nerr ? "FAIL" : "PASS");
  return nerr;
}

} // namespace marmot
} // namespace woodland
