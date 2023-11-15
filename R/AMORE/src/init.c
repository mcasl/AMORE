#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
 Check these declarations against the C/Fortran source code.
 */

/* .Call calls */
extern SEXP ADAPTgd_loop_MLPnet(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP ADAPTgdwm_loop_MLPnet(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP BATCHgd_loop_MLPnet(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP BATCHgdwm_loop_MLPnet(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP sim_Forward_MLPnet(SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"ADAPTgd_loop_MLPnet",   (DL_FUNC) &ADAPTgd_loop_MLPnet,   5},
  {"ADAPTgdwm_loop_MLPnet", (DL_FUNC) &ADAPTgdwm_loop_MLPnet, 5},
  {"BATCHgd_loop_MLPnet",   (DL_FUNC) &BATCHgd_loop_MLPnet,   6},
  {"BATCHgdwm_loop_MLPnet", (DL_FUNC) &BATCHgdwm_loop_MLPnet, 6},
  {"sim_Forward_MLPnet",    (DL_FUNC) &sim_Forward_MLPnet,    4},
  {NULL, NULL, 0}
};

void R_init_AMORE(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
