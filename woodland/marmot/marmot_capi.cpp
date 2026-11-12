#include "woodland/marmot/marmot_capi.h"
#include "woodland/marmot/greens_matrix.hpp"

#include <cstring>
#include <exception>

using woodland::marmot::MatrixEvaluator;
using woodland::marmot::MatrixOptions;

extern "C" {

void* woodland_marmot_create (const char* params, char* err, int errlen) {
  try {
    return new MatrixEvaluator(MatrixOptions::parse(params));
  } catch (const std::exception& ex) {
    if (err && errlen > 0) {
      std::strncpy(err, ex.what(), errlen - 1);
      err[errlen - 1] = '\0';
    }
    return nullptr;
  }
}

void woodland_marmot_destroy (void* e) {
  delete static_cast<MatrixEvaluator*>(e);
}

int woodland_marmot_ncell (const void* e) {
  return static_cast<const MatrixEvaluator*>(e)->get_ncell();
}

int woodland_marmot_entries (const void* e, int isrc, int ircv,
                             int disloc_comp, double sigma[6]) {
  try {
    static_cast<const MatrixEvaluator*>(e)->calc_entries(
      isrc, ircv, disloc_comp, sigma);
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}

int woodland_marmot_cell (const void* e, int ci, double xyz[3],
                          double lcs[9]) {
  const auto& me = *static_cast<const MatrixEvaluator*>(e);
  if (ci < 0 || ci >= me.get_ncell()) return 1;
  me.get_surface().cell_ctr_xyz(ci, xyz);
  me.get_surface().cell_ctr_lcs(ci, lcs);
  return 0;
}

} // extern "C"
