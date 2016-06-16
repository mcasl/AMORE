from libc.math import tanh, exp, cos, sin, fabs, atan, pi
from common cimport RealNumber

cdef class ActivationFunction:
    cdef public:
        str label
    cpdef RealNumber original(self, RealNumber induced_local_field)
    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output)

cdef class TanhActivationFunction(ActivationFunction):
    pass

cdef class IdentityActivationFunction(ActivationFunction):
    pass

cdef class ThresholdActivationFunction(ActivationFunction):
    pass

cdef class LogisticActivationFunction(ActivationFunction):
    pass

cdef class ExponentialActivationFunction(ActivationFunction):
    pass

cdef class ReciprocalActivationFunction(ActivationFunction):
    pass

cdef class SquareActivationFunction(ActivationFunction):
    pass

cdef class GaussActivationFunction(ActivationFunction):
    pass

cdef class ElliotActivationFunction(ActivationFunction):
    pass

cdef class SineActivationFunction(ActivationFunction):
    pass

cdef class CosineActivationFunction(ActivationFunction):
    pass

cdef class ArctanActivationFunction(ActivationFunction):
    pass
