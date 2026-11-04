// Yudong: 2026-10-14
#include "woodland/squirrel/matrix_zxy_capi.h"
#include "woodland/squirrel/ctzx.hpp"

#include <cstring>
#include <exception>

using woodland::squirrel::ctzx::MatrixZxyEvaluator;

extern "C" {

void* woodland_mzxy_create (const char* params, char* err, int errlen) {
  try {
    return new MatrixZxyEvaluator(params);
  } catch (const std::exception& ex) {
    if (err && errlen > 0) {
      std::strncpy(err, ex.what(), errlen - 1);
      err[errlen - 1] = '\0';
    }
    return nullptr;
  }
}

void woodland_mzxy_destroy (void* e) {
  delete static_cast<MatrixZxyEvaluator*>(e);
}

int woodland_mzxy_ncell (const void* e) {
  return static_cast<const MatrixZxyEvaluator*>(e)->get_ncell();
}

int woodland_mzxy_entries (const void* e, int isrc, int ircv, double a[2]) {
  static_assert(MatrixZxyEvaluator::ncomp == 2, "a has 2 entries");
  try {
    static_cast<const MatrixZxyEvaluator*>(e)->calc_entries(isrc, ircv, a);
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}

} // extern "C"
