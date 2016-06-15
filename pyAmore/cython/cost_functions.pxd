from common cimport RealNumber, f2param

cdef class CostFunction:
    cdef public:
        object original  # TODO: change
        object derivative  # TODO: change
