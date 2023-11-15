
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"


/**
##########################################################
#	Adaptative Gradient Descent (with momentum)
##########################################################
*/

SEXP ADAPTgdwm_loop_MLPnet (SEXP origNet, SEXP Ptrans, SEXP Ttrans, SEXP nepochs, SEXP rho) {
   int * Ptransdim, *Ttransdim, fila, columna, Pcounter, Tcounter;
   int considered_input, ind_neuron, ind_other_neuron, that_aim, ind_weight;
   double aux_DELTA, x_input, a, bias_change, weight_change;
   int epoch, n_epochs;

   SEXP R_fcall, args, arg1, arg2, arg3;
   SEXP net;
   struct AMOREneuron * ptneuron, * pt_that_neuron;
   struct AMOREnet * ptnet;


   PROTECT(net=duplicate(origNet));
   Ptransdim = INTEGER(coerceVector(getAttrib(Ptrans, R_DimSymbol), INTSXP));
   Ttransdim = INTEGER(coerceVector(getAttrib(Ttrans, R_DimSymbol), INTSXP));
   n_epochs  = INTEGER(nepochs)[0];

   ptnet = copynet_RC(net);
   for (epoch=0; epoch < n_epochs; epoch++) {
      for (fila=0, Pcounter=0, Tcounter=0; fila < Ptransdim[1]; fila++) {
         for( columna =0; columna < Ptransdim[0] ; columna++, Pcounter++) {
            ptnet->input[columna] =  REAL(Ptrans)[Pcounter];
         }
         for( columna =0; columna < Ttransdim[0] ; columna++, Tcounter++) {
            ptnet->target[columna] =  REAL(Ttrans)[Tcounter];
         }
         /** BEGIN   void adaptgdwm_forward_mlpnet(AMOREnet * ptnet)   */
         for (ind_neuron=0; ind_neuron <= ptnet->last_neuron ; ind_neuron++ ) {
            ptneuron = ptnet->neurons[ind_neuron];
            /* BEGIN adaptgdwm_forward_MLPneuron */
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
                  ptneuron->v1 =  b_tansig / a_tansig * (a_tansig - ptneuron->v0)*(a_tansig + ptneuron->v0);
                  break;
               case SIGMOID_ACTF:
                  ptneuron->v0 =  1/(1+exp(- a_sigmoid * a)) ; 
                  ptneuron->v1 =  a_sigmoid * ptneuron->v0 * ( 1 - ptneuron->v0 );
                  break;
               case PURELIN_ACTF:
                  ptneuron->v0 = a; 
                  ptneuron->v1 = 1;
                  break;
               case HARDLIM_ACTF:
                  if (a>=0) {
                     ptneuron->v0 = 1.0;
                  } else {
                     ptneuron->v0 = 0.0;
                  }
                  ptneuron->v1 = NA_REAL;
                 break;
               case CUSTOM_ACTF:
                  PROTECT(args    = allocVector(REALSXP,1));
                  REAL(args)[0]   = a;
                  PROTECT(R_fcall = lang2(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, ind_neuron), id_F0), args));
                  ptneuron->v0    = REAL(eval (R_fcall, rho))[0];
                  PROTECT(args    = allocVector(REALSXP,1));   
                  REAL(args)[0]   = a;
                  PROTECT(R_fcall = lang2(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, ind_neuron), id_F1), args));
                  ptneuron->v1    = REAL(eval (R_fcall, rho))[0];
                  UNPROTECT(4);
                  break; 
            }
         /* END adaptgdwm_forward_MLPneuron */
         }
         /** END     void adaptgdwm_forward_mlpnet(AMOREnet * ptnet)   */


         /* BEGIN   void adaptgdwm_backwards_MLPnet (AMOREnet * ptnet, SEXP rho) */
         for ( ind_neuron=ptnet->last_neuron; ind_neuron >=0;  ind_neuron-- ) {
            ptneuron=ptnet->neurons[ind_neuron];
         /**/
            if (ptneuron->type==TYPE_OUTPUT) {
               switch(ptnet->deltaE.name) {
                  case LMS_NAME:
                     aux_DELTA = ptneuron->v0 - ptnet->target[-1+ptneuron->output_aims[0]];
                  break;
                  case LMLS_NAME:
                     aux_DELTA = ptneuron->v0- ptnet->target[-1+ptneuron->output_aims[0]];
                     aux_DELTA = aux_DELTA / (1 + aux_DELTA*aux_DELTA / 2);
                     break;
                  default:   /** if (ptneuron->deltaE.name==TAO_NAME)   de momento tao es como custom*/ 
                    /** ####### OJO FALTA cambiar el TAO  */
                    PROTECT(args  = allocVector(VECSXP,3)     );
                    PROTECT(arg3  = net                       );
                    PROTECT(arg2  = allocVector(REALSXP,1)    );
                    PROTECT(arg1  = allocVector(REALSXP,1)    );
                    REAL(arg1)[0] = ptneuron->v0;
                    REAL(arg2)[0] =  ptnet->target[-1+ptneuron->output_aims[0]];
                    SET_VECTOR_ELT(args, 0, arg1);
                    SET_VECTOR_ELT(args, 1, arg2);
                    SET_VECTOR_ELT(args, 2, arg3);
                    PROTECT(R_fcall = lang2(DELTAE_F, args) );
                    aux_DELTA = REAL(eval (R_fcall, rho))[0];
                    UNPROTECT(5);
                    break;
               };
            } else {
               aux_DELTA = 0.0;
               for ( ind_other_neuron=0; ind_other_neuron <= ptneuron->last_output_link ; ind_other_neuron++ ) {
                  pt_that_neuron = ptneuron->output_links[ind_other_neuron];
                  that_aim       = -1+ptneuron->output_aims[ind_other_neuron];
                  aux_DELTA     += pt_that_neuron->method_dep_variables.adaptgdwm.delta * pt_that_neuron->weights[that_aim] ;
               }
            }
            ptneuron->method_dep_variables.adaptgdwm.delta = aux_DELTA * ptneuron->v1;
            bias_change = ptneuron->method_dep_variables.adaptgdwm.momentum  * ptneuron->method_dep_variables.adaptgdwm.former_bias_change - ptneuron->method_dep_variables.adaptgdwm.learning_rate * ptneuron->method_dep_variables.adaptgdwm.delta;
            ptneuron->bias += bias_change;
            ptneuron->method_dep_variables.adaptgdwm.former_bias_change = bias_change;
            for (ind_weight = 0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
               considered_input = ptneuron->input_links[ind_weight];
               if (considered_input < 0 ) {
                  x_input = ptnet->input[-1-considered_input];
               } else {
                  x_input = ptnet->neurons[-1+considered_input]->v0;
               }
               weight_change  =  ptneuron->method_dep_variables.adaptgdwm.momentum  * ptneuron->method_dep_variables.adaptgdwm.former_weight_change[ind_weight] - ptneuron->method_dep_variables.adaptgdwm.learning_rate * ptneuron->method_dep_variables.adaptgdwm.delta  * x_input ;
               ptneuron->weights[ind_weight] += weight_change;
               ptneuron->method_dep_variables.adaptgdwm.former_weight_change[ind_weight] = weight_change;
            }
            /**/
         }
         /* END    void adaptgdwm_backwards_MLPnet (AMOREnet * ptnet, SEXP rho) */
      }
   }
   copynet_CR (net, ptnet);
   UNPROTECT(1);
   return (net);
}




