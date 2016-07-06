# cython: profile=True

from libc.math cimport log
from common cimport RealNumber
import numpy as np
cimport numpy as np


cdef class CostFunction:
    def __init__(self):
        self.label = None

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual ** 2

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual

cdef class AdaptLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLms'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual ** 2

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        return residual

# TODO: revise for multiple outputs
cdef class AdaptLmLsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'AdaptLmLs'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        cdef RealNumber result = log(1 + residual ** 2 / 2)
        return result

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        cdef RealNumber result = residual / (1 + residual ** 2 / 2)
        return result


cdef class BatchLmsCostFunction(CostFunction):
    def __init__(self):
        self.label = 'BatchLms'

    cpdef RealNumber original(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        cdef RealNumber result = np.mean(residual ** 2)
        return result

    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target):
        cdef RealNumber residual = prediction - target
        cdef RealNumber result = np.mean(residual)
        return result



############################################################

cost_functions_set = {'adaptLMS':  AdaptLmsCostFunction(),
                      'adaptLMLS': AdaptLmLsCostFunction(),
                      'default': CostFunction(),
                      }
