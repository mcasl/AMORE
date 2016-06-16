# cython: profile=True

# from libc.math cimport tanh

from common cimport RealNumber
import numpy as np

#cimport numpy as np #: TODO: cimport


cdef class CostFunction:
    def __init__(self):
        self.label = None

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual ** 2

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return residual

cdef class AdaptLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLms'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual ** 2

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return residual


cdef class AdaptLmLsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLmLs'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        result = np.mean(np.log(1 + residual ** 2 / 2))
        return result

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        result = residual / (1 + residual ** 2 / 2)
        return result


cdef class BatchLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'BatchLms'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return np.mean(residual ** 2)

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        residual = prediction - target
        return np.mean(residual)



############################################################

cost_functions_set = {'adaptLMS':  AdaptLmsCostFunction(),
                      'adaptLMLS': AdaptLmLsCostFunction(),
                      'default': CostFunction(),
                      }
