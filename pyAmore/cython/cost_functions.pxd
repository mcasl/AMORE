from common cimport RealNumber

cdef class CostFunction:
    cdef public:
        object label

cdef class AdaptLmsCostFunction(CostFunction):
    cpdef RealNumber original(  AdaptLmsCostFunction self, RealNumber prediction, RealNumber target)
    cpdef RealNumber derivative(AdaptLmsCostFunction self, RealNumber prediction, RealNumber target)


cdef class AdaptLmLsCostFunction(CostFunction):
    cpdef RealNumber original(  AdaptLmLsCostFunction self, RealNumber prediction, RealNumber target)
    cpdef RealNumber derivative(AdaptLmLsCostFunction self, RealNumber prediction, RealNumber target)

cdef class BatchLmsCostFunction(CostFunction):
    cpdef RealNumber original(  BatchLmsCostFunction self, RealNumber prediction, RealNumber target)
    cpdef RealNumber derivative(BatchLmsCostFunction self, RealNumber prediction, RealNumber target)
