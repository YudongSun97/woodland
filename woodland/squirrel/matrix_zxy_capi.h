#ifndef INCLUDE_WOODLAND_SQUIRREL_MATRIX_ZXY_CAPI
#define INCLUDE_WOODLAND_SQUIRREL_MATRIX_ZXY_CAPI

/* Yudong: 2026-10-14
   C interface to woodland::squirrel::ctzx::MatrixZxyEvaluator, so that codes
   compiled with another C++ standard (hmmvp uses C++14 exception
   specifications) can evaluate woodland matrix entries without including the
   woodland headers. */

#ifdef __cplusplus
extern "C" {
#endif

/* params as for 'woodland_squirrel_unittest ct_matrix_zxy'. Returns NULL on
   error, with the message copied to err (if err != NULL). */
void* woodland_mzxy_create(const char* params, char* err, int errlen);
void woodland_mzxy_destroy(void* e);
int woodland_mzxy_ncell(const void* e);
/* a[0], a[1] = A_xz(ircv, isrc), A_zz(ircv, isrc) for surface=zxy, 0-based
   indices. Thread safe. Returns 0 on success. */
int woodland_mzxy_entries(const void* e, int isrc, int ircv, double a[2]);

#ifdef __cplusplus
}
#endif

#endif
