from common cimport *

cdef class CostFunction:
    cdef public:
        object original
        object derivative
