#include "woodland/acorn/openmp.hpp"
#include "woodland/acorn/util.hpp"
#include "woodland/acorn/dbg.hpp"

#include "woodland/squirrel/unittest.hpp"
#include "woodland/squirrel/ctzx.hpp"

#include <fstream>

struct Command {
  // ct_ab_zx: convergence test of a w.r.t. reference b, for extruded z(x)
  // surface. w is Woodland, o is Okada.
  // ct_stress_zxy: stress calculation on z=f(x,y) surface (no exact reference).
  // Yudong 03/04/2026: ct_stress_zxy_exact: Exact reference for z=f(x,y) surface.
  // ct_matrix_zxy: influence matrices on z=f(x,y) surface.
  enum Enum : int
    { unittest = 0, runcase,
      ct_we_zx, ct_oe_zx, ct_stress_zxy, ct_stress_zxy_exact, ct_matrix_zxy,
      invalid };
  static Enum convert(const int i);
  static Enum convert(const std::string& e);
  static std::string convert(const Enum e);
  static bool is_valid(const Enum e);
};

Command::Enum Command::convert (const int i) {
  auto s = static_cast<Command::Enum>(i);
  if ( ! is_valid(s)) s = invalid;
  return s;
}

Command::Enum Command::convert (const std::string& e) {
  if (e == "unittest") return unittest;
  if (e == "runcase") return runcase;
  if (e == "ct_we_zx") return ct_we_zx;
  if (e == "ct_oe_zx") return ct_oe_zx;
  if (e == "ct_stress_zxy") return ct_stress_zxy;
  // Yudong 03/04/2026
  if (e == "ct_stress_zxy_exact") return ct_stress_zxy_exact;
  if (e == "ct_matrix_zxy") return ct_matrix_zxy;
  return invalid;
}

std::string Command::convert (const Command::Enum e) {
  switch (e) {
  case unittest: return "unittest";
  case runcase: return "runcase";
  case ct_we_zx: return "ct_we_zx";
  case ct_oe_zx: return "ct_oe_zx";
  case ct_stress_zxy: return "ct_stress_zxy";
  // Yudong 03/04/2026
  case ct_stress_zxy_exact: return "ct_stress_zxy_exact";
  case ct_matrix_zxy: return "ct_matrix_zxy";
  case invalid:
  default: return "invalid";
  }
}

bool Command::is_valid (const Enum e) {
  return e >= unittest && e < invalid;
}

int main (int argc, char** argv) {
  using namespace woodland;
  using namespace squirrel;
  Command::Enum command = Command::unittest;
  if (argc > 1)
    command = Command::convert(argv[1]);
  if (command == Command::invalid) {
    printf("Invalid command: %s\n", argv[1]);
    printf("%s <command> args...\n", argv[0]);
    return -1;
  }
  printf("#threads %d\n", woodland::acorn::get_max_threads());
  switch (command) {
  case Command::unittest: return unittest();
  case Command::runcase:
  case Command::ct_we_zx:
  case Command::ct_oe_zx:
  case Command::ct_stress_zxy:
  // Yudong 03/04/2026
  case Command::ct_stress_zxy_exact:
  case Command::ct_matrix_zxy: {
    std::string params;
    if (command == Command::ct_stress_zxy ||
        command == Command::ct_stress_zxy_exact ||
        command == Command::ct_matrix_zxy)
      params = "nx=4,ny=4,srfrecon=0,testcase=20";
    else
      params = "testcase=0";
    if (argc > 2) params = argv[2];
    switch (command) {
    case Command::runcase: ctzx::run_case(params); break;
    case Command::ct_we_zx: ctzx::convtest_w_vs_e(params); break;
    case Command::ct_oe_zx: ctzx::convtest_o_vs_e(params); break;
    case Command::ct_stress_zxy: ctzx::run_stress_zxy(params); break;
    // Yudong 03/04/2026
    case Command::ct_stress_zxy_exact: ctzx::run_stress_zxy_exact(params); break;
    case Command::ct_matrix_zxy: ctzx::run_matrix_zxy(params); break;
    default: assert(0);
    }
  } break;
  case Command::invalid:
  default: {
    printf("Invalid command: %s\n", argv[1]);
    return -1;
  }
  }
  return 0;
}
