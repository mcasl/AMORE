
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"



struct AMOREnet * copynet_RC (SEXP net);
void              copynet_CR (SEXP net, struct AMOREnet * ptnet);
/**
##########################################################
# copynet_RC
#	Copies the SEXP net to the *ptnet
##########################################################
**/

struct AMOREnet * copynet_RC (SEXP net) {
   struct AMOREnet * ptnet;
   struct AMOREneuron * ptneuron;
   int i, ind_neuron, ind_input_neuron, ind_output_neuron, ind_layer;
   SEXP neuron;
   int aux_neuron;

   ptnet = (struct AMOREnet *) R_alloc(1, sizeof(struct AMOREnet));
   ptnet->last_neuron  = -1+LENGTH(NET_NEURONS);
   ptnet->last_input   = -1+LENGTH(NET_INPUT);
   ptnet->last_output  = -1+LENGTH(NET_OUTPUT);
   ptnet->input  = (double *) R_alloc(ptnet->last_input  + 1, sizeof(double));
   ptnet->output = (double *) R_alloc(ptnet->last_output + 1, sizeof(double));
   ptnet->target = (double *) R_alloc(ptnet->last_output + 1, sizeof(double));
   for (i=0; i <= ptnet->last_input; i++) {
      ptnet->input[i] =  REAL(NET_INPUT)[i];
   }
   for (i=0; i <= ptnet->last_output; i++) {
      ptnet->output[i] =  REAL(NET_OUTPUT)[i];
      ptnet->target[i] =  REAL(NET_OUTPUT)[i];
   }
   ptnet->deltaE.name = INTEGER(DELTAE_NAME)[0];
   ptnet->deltaE.stao = REAL(DELTAE_STAO)[0];

   ptnet->neurons = (struct AMOREneuron **) R_alloc(ptnet->last_neuron + 1, sizeof(struct AMOREneuron *));
   for (ind_neuron=0; ind_neuron <= ptnet->last_neuron; ind_neuron ++ ) {
      ptnet->neurons[ind_neuron] = (struct AMOREneuron *) R_alloc(1, sizeof(struct AMOREneuron));
   /* do not join with the following block*/
   }
   for (ind_neuron=0; ind_neuron <= ptnet->last_neuron; ind_neuron ++ ) {
      PROTECT(neuron=VECTOR_ELT(NET_NEURONS, ind_neuron ) );
      ptneuron = ptnet->neurons[ind_neuron];
      ptneuron->id               =  INTEGER(ID)[0];
      if (strcmp(CHAR(STRING_ELT(TYPE,0)),"output")==0) {
         ptneuron->type = TYPE_OUTPUT;
      } else {
         ptneuron->type = TYPE_HIDDEN;
      }

      ptneuron->actf             =  INTEGER(ACTIVATION_FUNCTION)[0] ;
      ptneuron->last_output_link =  -1 + LENGTH(OUTPUT_LINKS) ;
      ptneuron->last_input_link  =  -1 + LENGTH(INPUT_LINKS)  ;
      ptneuron->output_aims      = (int *) R_alloc(ptneuron->last_output_link+1, sizeof(int));
      ptneuron->input_links      = (int *) R_alloc(ptneuron->last_input_link+1,  sizeof(int));
      ptneuron->output_links     = (struct AMOREneuron **) R_alloc(ptneuron->last_output_link+1, sizeof(struct AMOREneuron *));
      ptneuron->weights          = (double *) R_alloc(ptneuron->last_input_link+1, sizeof(double));

      for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
         ptneuron->input_links[ind_input_neuron] = INTEGER(INPUT_LINKS)[ind_input_neuron];
         ptneuron->weights[ind_input_neuron] = REAL(WEIGHTS)[ind_input_neuron];
      }
      for (ind_output_neuron=0; ind_output_neuron <= ptneuron->last_output_link; ind_output_neuron++) {
         ptneuron->output_aims[ind_output_neuron]  = INTEGER(OUTPUT_AIMS)[0];
         if(INTEGER(OUTPUT_LINKS)[ind_output_neuron]==NA_INTEGER){
            ptneuron->output_links[ind_output_neuron] = NULL;
         } else {
            ptneuron->output_links[ind_output_neuron] = ptnet->neurons[-1+INTEGER(OUTPUT_LINKS)[ind_output_neuron]];
         }
      }
      ptneuron->bias    = REAL(BIAS)[0];
      ptneuron->v0      = REAL(V0)[0];
      ptneuron->v1      = REAL(V1)[0];
      if (strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgd")==0) {
         ptneuron->method  = METHOD_ADAPTgd;
         ptneuron->method_dep_variables.adaptgd.delta              = REAL(ADAPTgd_DELTA)[0] ;
         ptneuron->method_dep_variables.adaptgd.learning_rate      = REAL(ADAPTgd_LEARNING_RATE)[0] ;
      } else if (strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgdwm")==0) {
         ptneuron->method  = METHOD_ADAPTgdwm;
         ptneuron->method_dep_variables.adaptgdwm.delta              = REAL(ADAPTgdwm_DELTA)[0] ;
         ptneuron->method_dep_variables.adaptgdwm.learning_rate      = REAL(ADAPTgdwm_LEARNING_RATE)[0] ;
         ptneuron->method_dep_variables.adaptgdwm.momentum           = REAL(ADAPTgdwm_MOMENTUM)[0] ;
         ptneuron->method_dep_variables.adaptgdwm.former_bias_change = REAL(ADAPTgdwm_FORMER_BIAS_CHANGE)[0];
         ptneuron->method_dep_variables.adaptgdwm.former_weight_change = (double *) R_alloc(ptneuron->last_input_link+1, sizeof(double));
         for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
            ptneuron->method_dep_variables.adaptgdwm.former_weight_change[ind_input_neuron] = REAL(ADAPTgdwm_FORMER_WEIGHT_CHANGE)[ind_input_neuron] ;
         }
      } else if (strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgd")==0) {
         ptneuron->method  = METHOD_BATCHgd;
         ptneuron->method_dep_variables.batchgd.delta              = REAL(BATCHgd_DELTA)[0] ;
         ptneuron->method_dep_variables.batchgd.learning_rate      = REAL(BATCHgd_LEARNING_RATE)[0] ;
         ptneuron->method_dep_variables.batchgd.sum_delta_x        = (double *) R_alloc(ptneuron->last_input_link+1, sizeof(double));
         for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
            ptneuron->method_dep_variables.batchgd.sum_delta_x[ind_input_neuron] = REAL(BATCHgd_SUM_DELTA_X)[ind_input_neuron] ;
         }      
         ptneuron->method_dep_variables.batchgd.sum_delta_bias     = REAL(BATCHgd_SUM_DELTA_BIAS)[0] ;
      } else if (strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgdwm")==0) {
         ptneuron->method  = METHOD_BATCHgdwm;
         ptneuron->method_dep_variables.batchgdwm.delta            = REAL(BATCHgdwm_DELTA)[0] ;
         ptneuron->method_dep_variables.batchgdwm.learning_rate    = REAL(BATCHgdwm_LEARNING_RATE)[0] ;
         ptneuron->method_dep_variables.batchgdwm.sum_delta_x      = (double *) R_alloc(ptneuron->last_input_link+1, sizeof(double));
         for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
            ptneuron->method_dep_variables.batchgdwm.sum_delta_x[ind_input_neuron] = REAL(BATCHgdwm_SUM_DELTA_X)[ind_input_neuron] ;
         }      
         ptneuron->method_dep_variables.batchgdwm.sum_delta_bias     = REAL(BATCHgdwm_SUM_DELTA_BIAS)[0] ;
         ptneuron->method_dep_variables.batchgdwm.momentum           = REAL(BATCHgdwm_MOMENTUM)[0] ;
         ptneuron->method_dep_variables.batchgdwm.former_bias_change = REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0] ;
         ptneuron->method_dep_variables.batchgdwm.former_weight_change = (double *) R_alloc(ptneuron->last_input_link+1, sizeof(double));
         for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
            ptneuron->method_dep_variables.batchgdwm.former_weight_change[ind_input_neuron] = REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[ind_input_neuron] ;
         }
      }
      UNPROTECT(1);
   }
   ptnet->last_layer   = -2+LENGTH(NET_LAYERS); /* the first one doesn't count */
   ptnet->layer_size   = (int *) R_alloc(ptnet->last_layer  + 1, sizeof(int));
   ptnet->layers = (struct AMOREneuron ***) R_alloc(1+ptnet->last_layer, sizeof(struct AMOREneuron **));   
   for (ind_layer=0; ind_layer <= ptnet->last_layer ; ind_layer++) {
      ptnet->layer_size[ind_layer] = LENGTH(VECTOR_ELT(NET_LAYERS, 1+ind_layer));
      ptnet->layers[ind_layer] = (struct AMOREneuron **) R_alloc(ptnet->layer_size[ind_layer], sizeof(struct AMOREneuron *));
      for (ind_neuron=0; ind_neuron < ptnet->layer_size[ind_layer]; ind_neuron++) {
         aux_neuron = -1+INTEGER(VECTOR_ELT(NET_LAYERS, 1+ind_layer))[ind_neuron];
         ptnet->layers[ind_layer][ind_neuron] = ptnet->neurons[ aux_neuron ];
      }
   }
   return (ptnet);
}

/** 
################################
# copynet_CR
# Copies *ptnet to SEXP net
################################
**/
void copynet_CR (SEXP net, struct AMOREnet * ptnet){
   struct AMOREneuron * ptneuron;
   int ind_neuron, ind_input_neuron, ind_weight;
   SEXP neuron;

   REAL(DELTAE_STAO)[0] = ptnet->deltaE.stao ;

   for (ind_neuron=0; ind_neuron <= ptnet->last_neuron; ind_neuron ++ ) {
      PROTECT(neuron=VECTOR_ELT(NET_NEURONS, ind_neuron ) );
      ptneuron = ptnet->neurons[ind_neuron];
      for (ind_input_neuron=0; ind_input_neuron <= ptneuron->last_input_link; ind_input_neuron++) {
         REAL(WEIGHTS)[ind_input_neuron] = ptneuron->weights[ind_input_neuron] ;
      }
      REAL(BIAS)[0] = ptneuron->bias ;
      REAL(V0)[0]   = ptneuron->v0 ;
      REAL(V1)[0]   = ptneuron->v1 ;

      switch(ptneuron->method) {
         case METHOD_ADAPTgd :
            REAL(ADAPTgd_DELTA)[0]         = ptneuron->method_dep_variables.adaptgd.delta ;
            REAL(ADAPTgd_LEARNING_RATE)[0] = ptneuron->method_dep_variables.adaptgd.learning_rate;
            break;
         case METHOD_ADAPTgdwm:
            REAL(ADAPTgdwm_DELTA)[0]              = ptneuron->method_dep_variables.adaptgdwm.delta;
            REAL(ADAPTgdwm_LEARNING_RATE)[0]      = ptneuron->method_dep_variables.adaptgdwm.learning_rate ;
            REAL(ADAPTgdwm_MOMENTUM)[0]           = ptneuron->method_dep_variables.adaptgdwm.momentum  ;
            REAL(ADAPTgdwm_FORMER_BIAS_CHANGE)[0] = ptneuron->method_dep_variables.adaptgdwm.former_bias_change ;
            for  (ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
               REAL(ADAPTgdwm_FORMER_WEIGHT_CHANGE)[ind_weight] = ptneuron->method_dep_variables.adaptgdwm.former_weight_change[ind_weight];
            }
            break;
         case METHOD_BATCHgd:
            REAL(BATCHgd_DELTA)[0]                = ptneuron->method_dep_variables.batchgd.delta ;
            REAL(BATCHgd_LEARNING_RATE)[0]        = ptneuron->method_dep_variables.batchgd.learning_rate ;
            for  (ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
               REAL(BATCHgd_SUM_DELTA_X)[ind_weight] = ptneuron->method_dep_variables.batchgd.sum_delta_x[ind_weight];
            }
            REAL(BATCHgd_SUM_DELTA_BIAS)[0]       = ptneuron->method_dep_variables.batchgd.sum_delta_bias ;
            break;
        default:
            REAL(BATCHgdwm_DELTA)[0]              = ptneuron->method_dep_variables.batchgdwm.delta ;
            REAL(BATCHgdwm_LEARNING_RATE)[0]      = ptneuron->method_dep_variables.batchgdwm.learning_rate ;
            for  (ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
               REAL(BATCHgdwm_SUM_DELTA_X)[ind_weight] = ptneuron->method_dep_variables.batchgdwm.sum_delta_x[ind_weight];
            }            
            REAL(BATCHgdwm_SUM_DELTA_BIAS)[0]     = ptneuron->method_dep_variables.batchgdwm.sum_delta_bias;
            REAL(BATCHgdwm_MOMENTUM)[0]           = ptneuron->method_dep_variables.batchgdwm.momentum ;
            REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0] = ptneuron->method_dep_variables.batchgdwm.former_bias_change ;
            for (ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
                REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[ind_weight] = ptneuron->method_dep_variables.batchgdwm.former_weight_change[ind_weight];
            }
            break;
        }
      UNPROTECT(1);
   }
   return ;
}








