from common cimport *

import numpy as np

cdef class CostFunction:
    def __init__(self, cost_function_original, cost_function_derivative):
        self.original   = cost_function_original
        self.derivative = cost_function_derivative

    def __call__(self, prediction, target):
        return self.original(prediction, target)


############################################################
cpdef RealNumber adapt_lms_original(RealNumber prediction, RealNumber target):
    residual = prediction - target
    return residual ** 2


cpdef RealNumber adapt_lms_derivative(RealNumber prediction, RealNumber target):
    residual = prediction - target
    return residual

############################################################
cpdef RealNumber adapt_lmls_original(RealNumber prediction, RealNumber target):
    residual = prediction - target
    result = np.mean(np.log(1 + residual ** 2 / 2))
    return result

cpdef RealNumber adapt_lmls_derivative(RealNumber prediction, RealNumber target):
    residual = prediction - target
    result = residual / (1 + residual ** 2 / 2)
    return result
############################################################

cost_functions_set = {'adaptLMS':  CostFunction(adapt_lms_original,  adapt_lms_derivative),
                      'adaptLMLS': CostFunction(adapt_lmls_original, adapt_lmls_derivative),
                      'default':   CostFunction(adapt_lms_original,  adapt_lms_derivative)
                      }
