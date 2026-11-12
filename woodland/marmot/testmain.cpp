#include "woodland/acorn/openmp.hpp"

#include "woodland/marmot/unittest.hpp"
#include "woodland/marmot/greens_matrix.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

static void usage (const char* prog) {
  printf("%s [command [params]]\n"
         "  unittest (default)\n"
         "  matrix 'geom=FILE[,nx=,ny=,lcs=,lam=,mu=,halfspace=,dislocorder=,"
         "disloc=,comps=,out=]'\n"
         "  mesh 'geom=FILE[,nx=,ny=,lcs=,out=]'\n"
         "  stress 'geom=FILE[,slip=a:b:c, other matrix options]'\n"
         "  example_geom '[n=ni:nj,out=FILE]'\n"
         "See woodland/marmot/README.md.\n", prog);
}

int main (int argc, char** argv) {
  using namespace woodland::marmot;
  const std::string command = argc > 1 ? argv[1] : "unittest";
  const std::string params = argc > 2 ? argv[2] : "";
  if (command == "-h" || command == "--help" || command == "help") {
    usage(argv[0]);
    return 0;
  }
  printf("#threads %d\n", woodland::acorn::get_max_threads());
  try {
    if (command == "unittest") return unittest();
    if (command == "matrix") run_matrix(params);
    else if (command == "mesh") run_mesh(params);
    else if (command == "stress") run_stress(params);
    else if (command == "example_geom") run_example_geom(params);
    else {
      printf("Invalid command: %s\n", command.c_str());
      usage(argv[0]);
      return -1;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "%s: error: %s\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
