from common cimport *

cdef class CostFunction:
    cpdef  RealNumber cost_function(CostFunction self, RealNumber prediction, RealNumber target)
    cpdef  RealNumber derivative(CostFunction self, RealNumber prediction, RealNumber target)

cdef class AdaptLMS(CostFunction):
    pass

cdef class AdaptLMLS(CostFunction):
    pass
