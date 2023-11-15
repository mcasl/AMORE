/**
##############################################################
# batchgdwm ( BATCH gradient descent WITH momentum )
##############################################################
*/

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

SEXP BATCHgdwm_loop_MLPnet (SEXP origNet, SEXP Ptrans, SEXP Ttrans, SEXP nepochs, SEXP rho, SEXP thread_number ) {
   //The only difference between wm and without it is the weight update (and the place the values are stored, one is batchgd the other is batchgdwm)
   SEXP net;
   SEXP R_fcall, args, arg1, arg2, arg3;

   PROTECT(net=duplicate(origNet));
   int* Ptransdim = INTEGER(coerceVector(getAttrib(Ptrans, R_DimSymbol), INTSXP));
   int* Ttransdim = INTEGER(coerceVector(getAttrib(Ttrans, R_DimSymbol), INTSXP));
   int n_epochs  = INTEGER(nepochs)[0];
   struct AMOREnet* ptnet = copynet_RC(net);
   struct AMOREneuron** neurons = ptnet->neurons;

   /////////////////////////////////////////////////////////////////////////
   //Convert input and target to double only once (and instead of copying it every time, just change the pointers)
   //Different rows for easy switching pointers
   double*  input_data  = REAL(Ptrans);
   double*  target_data = REAL(Ttrans);
   double** inputs  = (double**) R_alloc(Ptransdim[1],sizeof(double*)); //This is an 'Index'
   double** targets = (double**) R_alloc(Ptransdim[1],sizeof(double*)); //This is an 'Index'

   for (int fila=0; fila < Ptransdim[1]; fila++) {
      inputs[fila]  = &input_data [fila*Ptransdim[0]];
      targets[fila] = &target_data[fila*Ttransdim[0]];
   }
   /////////////////////////////////////////////////////////////////////////

   /////////////////////////////////////////////////////////////////////////
   // Thread number calculation
   int n_threads = 1;
#ifdef _OPENMP
   {
      int max_threads = omp_get_max_threads();
      int given_threads = 0;

      if (isInteger(thread_number))
        given_threads = INTEGER(thread_number)[0];
      else if (isNumeric(thread_number))
        given_threads = floor(REAL(thread_number)[0]);

      if (given_threads <1) //I HAVE THE POWER TO SCHEDULE!
        if(max_threads  >1)
          n_threads = max_threads-1; //Leave a CPU free
        else
          n_threads = 1;
      else if (given_threads > max_threads)
        n_threads = max_threads;
      else
        n_threads = given_threads;

      if (neurons[0]->actf == CUSTOM_ACTF) //OMP + R custom functions = bad idea
        n_threads = 1;
      else if ((ptnet->deltaE.name != LMLS_NAME) && (ptnet->deltaE.name != LMS_NAME))
        n_threads = 1;

      //printf("Using %i threads from a max of %i.\n",n_threads ,max_threads);
   }
#endif
   /////////////////////////////////////////////////////////////////////////

   /////////////////////////////////////////////////////////////////////////
   //Contribution (who is to blame) : Parallelization done by Jose Maria
   //Memory allocation for running different threads in parallel:
   // Each thread will have his own pool of memory to handle the two kinds of temp vars:
   //   Vars used only inside the forwards/backwards (v0, v1 and method_delta)
   //     These vars will be initialized and read only by each thread
   //   Vars that hold the information on how much the weights and the bias should change 
   //     These vars will be initialized by each thread, then accumulated and read by the master thread when the batch is finished
   int n_neurons = ptnet->last_neuron+1;
   //Temp values, internal in each iteration
   double **  v0s                 = (double** ) R_alloc(n_threads,sizeof(double* )); //This is an 'Index'
   double **  v1s                 = (double** ) R_alloc(n_threads,sizeof(double* )); //This is an 'Index'
   double **  method_deltas       = (double** ) R_alloc(n_threads,sizeof(double* )); //This is an 'Index'
   //Accumulated values
   double **  method_deltas_bias  = (double** ) R_alloc(n_threads,sizeof(double* )); //This is an 'Index'
   double *** method_sums_delta_x = (double***) R_alloc(n_threads,sizeof(double**)); //This is an 'Index'

   for(int id_thread=0; id_thread<n_threads;id_thread++){
      double* chunk = (double*) R_alloc(4*n_neurons,sizeof(double)); //Actual chunk of memory for each thread, trying to avoid R_alloc calls
      //Advantages: Good proximity reference in cache for values of the same thread, and since it has at least 2 neurons
      // (Who would have a NNetwork with less than 2 neurons?), chunks are larger than 64 bytes (i7 L2 cache block size?)
      v0s               [id_thread] =  chunk             ;  
      v1s               [id_thread] = &chunk[  n_neurons];
      method_deltas     [id_thread] = &chunk[2*n_neurons];
      method_deltas_bias[id_thread] = &chunk[3*n_neurons];
      
      method_sums_delta_x[id_thread] = (double**) R_alloc(n_neurons,sizeof(double*)); //This is an 'Index'
      for(int i=0; i<n_neurons; i++) //Different weigth number for each layer, TODO: R_alloc each layer instead of each neuron
         method_sums_delta_x[id_thread][i] = (double*) R_alloc(neurons[i]->last_input_link+1,sizeof(double));
   }
   /////////////////////////////////////////////////////////////////////////

   /////////////////////////////////////////////////////////////////////////
   //Consistency (leave pnet as if the function had worked with their values instead of external ones)
   // R_alloc should handle freeing the memory, so it's not needed to free the previously allocated memory to avoid memory leaks
   // Changing pointer instead of copying data
   ptnet->input  = inputs[Ptransdim[1]-1];
   ptnet->target = targets[Ptransdim[1]-1];
   /////////////////////////////////////////////////////////////////////////
   
   /////////////////////////////////////////////////////////////////////////
   // Dividing learning rate and momentum by the number of samples in the training batch
   // Using local temp memory because of cache (proximity references) and direct access to memory and avoiding modification of header file
   // Using R_alloc for R to manage the memory
   double * neuron_learning_rate = (double*) R_alloc(n_neurons,sizeof(double));
   double * neuron_momentum      = (double*) R_alloc(n_neurons,sizeof(double));
   for(int i=0; i<n_neurons; i++){
      neuron_learning_rate[i] = ptnet->neurons[i]->method_dep_variables.batchgdwm.learning_rate / Ptransdim[1];
      neuron_momentum[i]      = ptnet->neurons[i]->method_dep_variables.batchgdwm.momentum      / Ptransdim[1];
   }
   /////////////////////////////////////////////////////////////////////////

   for (int epoch=0; epoch < n_epochs; epoch++) {
      //Run BATCH in parallel
      #pragma omp parallel num_threads(n_threads)
      {
#ifdef _OPENMP
        int id_thread = omp_get_thread_num();
#else
        int id_thread = 0;
#endif
        //////////////////////////////////////////////////////////////////////////////////////
        //// Using 'private' memory for each thread temp values instead of ptnet's own memory
        //// It's needed for multithreaded execution, in single thread model it's also used (is only modified if not compiled with OMP).
        //////////////////////////////////////////////////////////////////////////////////////
        //Select vars for this thread from the "memory pool":
        //  Used only by each thread:
        double* v0 = v0s[id_thread]; // double[n_neurons] //Using this instead of ptneuron->v0
        double* v1 = v1s[id_thread]; // double[n_neurons] //Using this instead of ptneuron->v1
        double* method_delta      = method_deltas[id_thread]; // double[n_neurons] //Using this instead of ptneuron->ptneuron->method_dep_variables.batchgdwm.delta
#ifdef _OPENMP
        //  Used to update weigths:
        double* method_delta_bias = method_deltas_bias[id_thread]; // double[n_neurons] //Instead of ptneuron->method_dep_variables.batchgdwm.sum_delta_bias
        double** method_sum_delta_x = method_sums_delta_x[id_thread]; // double*[n_neurons] //Instead of ptneuron->method_dep_variables.batchgdwm.sum_delta_x
        
        //Initialize vars that handle comm between batch execution and weight update
        for (int ind_neuron=0; ind_neuron <= ptnet->last_neuron; ind_neuron++){
            method_delta_bias[ind_neuron] = 0.0; //TODO: Should memset be used?
            for (int ind_weight=0; ind_weight <= neurons[ind_neuron]->last_input_link; ind_weight++)
              method_sum_delta_x[ind_neuron][ind_weight] = 0.0; //TODO: Should memset be used?
        }
#endif
        //////////////////////////////////////////////////////////////////////////////////////

        #pragma omp for 
        for (int fila=0; fila < Ptransdim[1]; fila++) {
           // R_alloc should handle freeing the memory, so it's not needed to free the previously allocated memory to avoid memory leaks
           // Also, these are read-only from this point onwards, should not be a problem accessing them on parallel threads 
           // ptnet->input  = inputs[fila];  //Moved into actual access
           // ptnet->target = targets[fila]; //Moved into actual access
           
           /* BEGIN   void batchgd_forward_mlpnet(AMOREnet * ptnet)   */
           for (int ind_neuron=0; ind_neuron <= ptnet->last_neuron ; ind_neuron++ ) {
              struct AMOREneuron * ptneuron = neurons[ind_neuron];
              /* BEGIN batchgd_forward_MLPneuron */
              double a=0.0;
              for (int ind_weight=0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
                 int considered_input = ptneuron->input_links[ind_weight];
                 double x_input = (considered_input < 0 )? inputs[fila][-1-considered_input] :  v0[-1+considered_input];
                 a +=  ptneuron->weights[ind_weight] * x_input;
              }
              a += ptneuron->bias;
              switch (ptneuron->actf) {
                 case TANSIG_ACTF:
                    v0[ind_neuron] =  a_tansig * tanh(a * b_tansig); 
                    v1[ind_neuron] =  b_tansig / a_tansig * (a_tansig - v0[ind_neuron])*(a_tansig + v0[ind_neuron]);
                    break;
                 case SIGMOID_ACTF:
                    v0[ind_neuron] =  1/(1+exp(- a_sigmoid * a)) ; 
                    v1[ind_neuron] =  a_sigmoid * v0[ind_neuron] * ( 1 - v0[ind_neuron] );
                    break;
                 case PURELIN_ACTF:
                    v0[ind_neuron] = a; 
                    v1[ind_neuron] = 1;
                    break;
                 case HARDLIM_ACTF:
                    if (a>=0) {
                       v0[ind_neuron] = 1.0;
                    } else {
                       v0[ind_neuron] = 0.0;
                    }
                    v1[ind_neuron] = NA_REAL;
                    break;
                 case CUSTOM_ACTF:
                    PROTECT(args    = allocVector(REALSXP,1));
                    REAL(args)[0]   = a;
                    PROTECT(R_fcall = lang2(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, ind_neuron), id_F0), args));
                    v0[ind_neuron]  = REAL(eval (R_fcall, rho))[0];
                    PROTECT(args    = allocVector(REALSXP,1));   
                    REAL(args)[0]   = a;
                    PROTECT(R_fcall = lang2(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, ind_neuron), id_F1), args));
                    v1[ind_neuron]  = REAL(eval (R_fcall, rho))[0];
                    UNPROTECT(4);
                    break; 
              }
           /* END batchgd_forward_MLPneuron */
           }
           /* END     void batchgd_forward_mlpnet(AMOREnet * ptnet)   */


           /* BEGIN   void Parcial_batchgd_backwards_MLPnet (AMOREnet * ptnet, SEXP rho) */
           for (int ind_neuron=ptnet->last_neuron; ind_neuron >=0;  ind_neuron-- ) {
              struct AMOREneuron* ptneuron=ptnet->neurons[ind_neuron];
           /**/
              double aux_DELTA = 0.0;
              if (ptneuron->type==TYPE_OUTPUT) {
                 switch(ptnet->deltaE.name) {
                    case LMS_NAME:
                       aux_DELTA = v0[ind_neuron] - targets[fila][-1+ptneuron->output_aims[0]];
                    break;
                    case LMLS_NAME:
                       aux_DELTA = v0[ind_neuron] - targets[fila][-1+ptneuron->output_aims[0]];
                       aux_DELTA = aux_DELTA / (1 + aux_DELTA*aux_DELTA / 2);
                       break;
                    default:   /* if (ptneuron->deltaE.name==TAO_NAME)   de momento tao es como custom*/ 
                      /* ####### OJO FALTA cambiar el TAO  */
                      PROTECT(args  = allocVector(VECSXP,3)     );
                      PROTECT(arg3  = net                       );
                      PROTECT(arg2  = allocVector(REALSXP,1)    );
                      PROTECT(arg1  = allocVector(REALSXP,1)    );
                      REAL(arg1)[0] = v0[ind_neuron];
                      REAL(arg2)[0] =  targets[fila][-1+ptneuron->output_aims[0]];
                      SET_VECTOR_ELT(args, 0, arg1);
                      SET_VECTOR_ELT(args, 1, arg2);
                      SET_VECTOR_ELT(args, 2, arg3);
                      PROTECT(R_fcall = lang2(DELTAE_F, args) );
                      aux_DELTA = REAL(eval (R_fcall, rho))[0];
                      UNPROTECT(5);
                      break;
                 };
              } else {
                 for (int ind_other_neuron=0; ind_other_neuron <= ptneuron->last_output_link ; ind_other_neuron++ ) {
                    struct AMOREneuron* pt_that_neuron = ptneuron->output_links[ind_other_neuron];
                    int that_aim       = -1+ptneuron->output_aims[ind_other_neuron];
                    aux_DELTA     += method_delta[pt_that_neuron->id-1] * pt_that_neuron->weights[that_aim] ;
                 }
              }

              method_delta[ptneuron->id-1] = aux_DELTA * v1[ind_neuron]; //R ids start in 1

              for (int ind_weight = 0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
                 int considered_input = ptneuron->input_links[ind_weight];
                 double x_input = considered_input < 0 ? inputs[fila][-1-considered_input] : v0[-1+considered_input];
#ifdef _OPENMP
                 method_sum_delta_x[ind_neuron][ind_weight] += method_delta[ptneuron->id-1] * x_input ;
              }
              method_delta_bias[ind_neuron] += method_delta[ptneuron->id-1];
           } /*/ End parcial backwards*/
        } /* end bucle fila */
      } //End parallel region

//Up to this point BATCHGD and BATCHGDWM are the same

      //////////////////////////////////////////////////////////////////////////////////////
      //Update ptnet with the values from batch calculations
      for(int id_thread=0; id_thread<n_threads;id_thread++){ //Maybe reduction could be used
        for (int ind_neuron=0; ind_neuron <= ptnet->last_neuron ; ind_neuron++ ) {
          struct AMOREneuron *  ptneuron = neurons[ind_neuron];
          ptneuron->method_dep_variables.batchgdwm.sum_delta_bias +=  method_deltas_bias[id_thread][ind_neuron];
          for (int ind_weight = 0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
            ptneuron->method_dep_variables.batchgdwm.sum_delta_x[ind_weight] += method_sums_delta_x[id_thread][ind_neuron][ind_weight];
          }
        }
      }
      //////////////////////////////////////////////////////////////////////////////////////
#else
                 ptneuron->method_dep_variables.batchgdwm.sum_delta_x[ind_weight] += method_delta[ptneuron->id-1] * x_input ;
              }
              ptneuron->method_dep_variables.batchgdwm.sum_delta_bias += method_delta[ptneuron->id-1];

           } /*/ End parcial backwards*/
        } /* end bucle fila */
      } //End parallel region (#pragma should have been ignored)
#endif     

      /** BEGIN UPDATEWEIGHTS */
      for (int ind_neuron=0; ind_neuron <= ptnet->last_neuron ; ind_neuron++ ) {
         struct AMOREneuron * ptneuron = ptnet->neurons[ind_neuron];
         double bias_change = neuron_momentum[ind_neuron] * ptneuron->method_dep_variables.batchgdwm.former_bias_change - neuron_learning_rate[ind_neuron] * ptneuron->method_dep_variables.batchgdwm.sum_delta_bias;
         ptneuron->method_dep_variables.batchgdwm.former_bias_change = bias_change ;
         ptneuron->bias += bias_change;
         for (int ind_weight = 0; ind_weight <= ptneuron->last_input_link; ind_weight++) {
            double weight_change  =  neuron_momentum[ind_neuron] * ptneuron->method_dep_variables.batchgdwm.former_weight_change[ind_weight] - neuron_learning_rate[ind_neuron] * ptneuron->method_dep_variables.batchgdwm.sum_delta_x[ind_weight] ;
            ptneuron->method_dep_variables.batchgdwm.former_weight_change[ind_weight] = weight_change ;
            ptneuron->weights[ind_weight] += weight_change;
         }
            /**/
      }
         /* END UPDATE WEIGHTS  */
   } /* end epoch loop*/
   copynet_CR (net, ptnet);
   UNPROTECT(1);
   return (net);
}
   

