#ifndef INCLUDE_WOODLAND_MARMOT_CAPI
#define INCLUDE_WOODLAND_MARMOT_CAPI

/* C interface to woodland::marmot::MatrixEvaluator, for codes (e.g., hmmvp)
   that cannot include the woodland headers. */

#ifdef __cplusplus
extern "C" {
#endif

/* params as for 'woodland_marmot_unittest matrix' (out= is ignored). Returns
   NULL on error, with the message copied to err (if err != NULL). */
void* woodland_marmot_create(const char* params, char* err, int errlen);
void woodland_marmot_destroy(void* e);
int woodland_marmot_ncell(const void* e);
/* sigma[0:6] = (xx, xy, xz, yy, yz, zz) in the receiver local frame at the
   center of cell ircv due to the unit dislocation component disloc_comp (0, 1,
   2) at cell isrc; 0-based indices. Thread safe. Returns 0 on success. */
int woodland_marmot_entries(const void* e, int isrc, int ircv,
                            int disloc_comp, double sigma[6]);
/* Cell center xyz[0:3] and local frame lcs[0:9] = (xhat, yhat, zhat). */
int woodland_marmot_cell(const void* e, int ci, double xyz[3],
                         double lcs[9]);

#ifdef __cplusplus
}
#endif

#endif
