from common cimport RealNumber

cdef class CostFunction:
    cdef public:
        object original  # TODO: change
        object derivative  # TODO: change
