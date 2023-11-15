from common cimport RealNumber

cdef class CostFunction:
    cdef public:
        str label
    cpdef RealNumber original(self, RealNumber prediction, RealNumber target)
    cpdef RealNumber derivative(self, RealNumber prediction, RealNumber target)

cdef class AdaptLmsCostFunction(CostFunction):
    pass

cdef class AdaptLmLsCostFunction(CostFunction):
    pass

cdef class BatchLmsCostFunction(CostFunction):
    pass
