#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"

/******************************************************************************************************************/
SEXP sim_Forward_MLPnet (SEXP net, SEXP Ptrans, SEXP ytrans, SEXP rho) {
   int * Ptransdim, *ytransdim, fila, columna, Pcounter, ycounter;
   int considered_input, ind_neuron, ind_weight;
   double  x_input, a;

   SEXP R_fcall, args;
   struct AMOREneuron * ptneuron;
   struct AMOREnet * ptnet;


   Ptransdim = INTEGER(coerceVector(getAttrib(Ptrans, R_DimSymbol), INTSXP));
   ytransdim = INTEGER(coerceVector(getAttrib(ytrans, R_DimSymbol), INTSXP));


   ptnet = copynet_RC(net);

   for (fila=0, Pcounter=0, ycounter=0; fila < Ptransdim[1]; fila++) {
      for( columna =0; columna < Ptransdim[0] ; columna++, Pcounter++) {
         ptnet->input[columna] =  REAL(Ptrans)[Pcounter];
      }
      for (ind_neuron=0; ind_neuron <= ptnet->last_neuron ; ind_neuron++ ) {
         ptneuron = ptnet->neurons[ind_neuron];
         for (a=0.0, ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
            considered_input = ptneuron->input_links[ind_weight];
            if (considered_input < 0 ) {
               x_input = ptnet->input[-1-considered_input];
            } else {
               x_input = ptnet->neurons[-1+considered_input]->v0;
            }
            a +=  ptneuron->weights[ind_weight] * x_input;
         }
         a += ptneuron->bias;
         switch (ptneuron->actf) {
            case TANSIG_ACTF:
               ptneuron->v0 =  a_tansig * tanh(a * b_tansig); 
               break;
            case SIGMOID_ACTF:
               ptneuron->v0 =  1/(1+exp(- a_sigmoid * a)) ; 
               break;
            case PURELIN_ACTF:
               ptneuron->v0 = a; 
               break;
            case HARDLIM_ACTF:
               if (a>=0) {
                  ptneuron->v0 = 1.0;
               } else {
                  ptneuron->v0 = 0.0;
               }
               break;
            case CUSTOM_ACTF:
               PROTECT(args    = allocVector(REALSXP,1));
               REAL(args)[0]   = a;
               PROTECT(R_fcall = lang2(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, ind_neuron), id_F0), args));
               ptneuron->v0    = REAL(eval (R_fcall, rho))[0];
               UNPROTECT(2);
             break; 
         }
      }
      for (ind_neuron=0; ind_neuron < ytransdim[0] ; ind_neuron++ ) {
         REAL(ytrans)[ycounter++] = ptnet->layers[ptnet->last_layer][ind_neuron]->v0;
      }
    } 

    return (ytrans);
}



/******************************************************************************************************************/

void print_MLPneuron (SEXP neuron) {
int i;
   Rprintf("***********************************************************\n");
/* ID */
   Rprintf("ID:\t\t\t%d \n",             INTEGER(ID)[0]            );
/* TYPE */
   Rprintf("TYPE:\t\t\t%s \n",           CHAR(STRING_ELT(TYPE,0))  );
/* ACTIVATION FUNCTION */
   Rprintf("ACT. FUNCTION:\t\t%s\n",     CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)) );
/* OUTPUT LINKS */
   if (INTEGER(OUTPUT_LINKS)[0] != NA_INTEGER ) {
      for (i=0; i<LENGTH(OUTPUT_LINKS); i++) {
         Rprintf("OUTPUT_LINKS %d:\t\t%d \n", i+1, INTEGER(OUTPUT_LINKS)[i]  );
     }
   } else {
      Rprintf("OUTPUT_LINKS:\t\tNA\n");
   }
/* OUTPUT AIMS */
   for (i=0; i<LENGTH(OUTPUT_AIMS); i++) {
      Rprintf("OUTPUT_AIMS.%d:\t\t%d \n", i+1, INTEGER(OUTPUT_AIMS)[i]   );
   }
/* INPUT LINKS */
   for (i=0; i<LENGTH(INPUT_LINKS); i++) {
      Rprintf("INPUT_LINKS.%d:\t\t%d \n", i+1, INTEGER(INPUT_LINKS)[i]  );
   }
/* WEIGHTS */
   for (i=0; i<LENGTH(WEIGHTS); i++) {
      Rprintf("WEIGHTS.%d:\t\t%f \n", i+1, REAL(WEIGHTS)[i]  );
   }
/* BIAS */
   Rprintf("BIAS:\t\t\t%f \n", REAL(BIAS)[0]  );
/* V0 */
   Rprintf("V0:\t\t\t%f \n", REAL(V0)[0]  );
/* V1 */
   Rprintf("V1:\t\t\t%f \n", REAL(V1)[0]  );
/* METHOD */
   Rprintf("METHOD:\t\t\t%s\n", CHAR(STRING_ELT(METHOD,0))  );
   Rprintf("METHOD DEP VARIABLES:\n");
   if (           strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgd"  )==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(ADAPTgd_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(ADAPTgd_LEARNING_RATE)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgdwm")==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(ADAPTgdwm_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(ADAPTgdwm_LEARNING_RATE)[0]  );
      /* MOMENTUM */
           Rprintf("MOMENTUM:\t\t%f \n",      REAL(ADAPTgdwm_MOMENTUM)[0]  );
      /* FORMER WEIGHT CHANGE */
           for (i=0; i<LENGTH(ADAPTgdwm_FORMER_WEIGHT_CHANGE); i++) {
              Rprintf("FORMER_WEIGHT_CHANGE.%d:\t%f \n", i+1,  REAL(ADAPTgdwm_FORMER_WEIGHT_CHANGE)[i]  );
           }
      /* FORMER BIAS CHANGE */
           Rprintf("FORMER_BIAS_CHANGE:\t%f \n", REAL(ADAPTgdwm_FORMER_BIAS_CHANGE)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgd"  )==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(BATCHgd_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(BATCHgd_LEARNING_RATE)[0]  );
      /* SUM DELTA X */
           for (i=0; i<LENGTH(BATCHgdwm_SUM_DELTA_X); i++) {
              Rprintf("SUM DELTA X %d:\t\t%f \n", i+1,  REAL(BATCHgd_SUM_DELTA_X)[i]  );
           }
      /* SUM DELTA BIAS */
           Rprintf("SUM DELTA BIAS:\t\t%f \n",REAL(BATCHgd_SUM_DELTA_BIAS)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgdwm")==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(BATCHgdwm_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(BATCHgdwm_LEARNING_RATE)[0]  );
      /* MOMENTUM */
           Rprintf("MOMENTUM:\t\t%f \n",      REAL(BATCHgdwm_MOMENTUM)[0]  );
      /* FORMER WEIGHT CHANGE */
           for (i=0; i<LENGTH(ADAPTgdwm_FORMER_WEIGHT_CHANGE); i++) {
              Rprintf("FORMER_WEIGHT_CHANGE.%d:\t%f \n", i+1,  REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[i]  );
           }
      /* FORMER BIAS CHANGE */
           Rprintf("FORMER_BIAS_CHANGE:\t%f \n", REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0]  );
      /* SUM DELTA X */
           for (i=0; i<LENGTH(BATCHgdwm_SUM_DELTA_X); i++) {
              Rprintf("SUM DELTA X %d:\t\t%f \n", i+1,  REAL(BATCHgdwm_SUM_DELTA_X)[i]  );
           }
      /* SUM DELTA BIAS */
           Rprintf("SUM DELTA BIAS:\t\t%f \n",REAL(BATCHgdwm_SUM_DELTA_BIAS)[0]  );
           Rprintf("***********************************************************\n");
   }

}
/******************************************************************************************************************/

