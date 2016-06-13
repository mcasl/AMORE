import numpy as np

cdef class CostFunction:
    cpdef  RealNumber cost_function(CostFunction self, RealNumber prediction, RealNumber target):
        return 0

    cpdef  RealNumber derivative(CostFunction self, RealNumber prediction, RealNumber target):
        return 0

    def __call__(self, prediction, target):
        return self.cost_function(prediction, target)

############################################################

cdef class AdaptLMS(CostFunction):
    cpdef  RealNumber cost_function(AdaptLMS, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return residual ** 2

    cpdef  RealNumber derivative(AdaptLMS self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return residual

############################################################
############################################################
"""
cpdef __batch_lms(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual ** 2)

cpdef __batch_lms_derivative(prediction: np.ndarray, target: np.ndarray):
    residual = prediction - target
    return np.mean(residual)
"""
############################################################
cdef class AdaptLMLS(CostFunction):
    cpdef  RealNumber cost_function(AdaptLMLS self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        result = np.mean(np.log(1 + residual ** 2 / 2))
        return result

    cpdef  RealNumber derivative(AdaptLMLS self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        result = residual / (1 + residual ** 2 / 2)
        return result
############################################################

cost_functions_set = {'adaptLMS': AdaptLMS(),
                      'adaptLMLS': AdaptLMLS(),
                      'default': AdaptLMS()
                      }
