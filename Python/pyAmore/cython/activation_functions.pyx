# cython: profile=True

from libc.math cimport tanh, exp, cos, sin, fabs, atan
from common cimport RealNumber

cdef class ActivationFunction:
    def __init__(self):
        self.label = 'default'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return tanh(induced_local_field)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 1.0 - neuron_output ** 2

#----------------------------------------------------------------- tanh ------------------------------------
cdef class TanhActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'tanh'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return tanh(induced_local_field)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 1.0 - neuron_output ** 2

#----------------------------------------------------------------- identity ------------------------------------
cdef class IdentityActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'identity'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return induced_local_field

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 1.0

#----------------------------------------------------------------- threshold ------------------------------------
cdef class ThresholdActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'threshold'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return 1.0 if induced_local_field > 0.0 else 0.0

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 0.0

#----------------------------------------------------------------- logistic ------------------------------------
cdef class LogisticActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'logistic'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return 1.0 / (1.0 + exp(-induced_local_field))

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return neuron_output * (1 - neuron_output)

#----------------------------------------------------------------- exponential ------------------------------------
cdef class ExponentialActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'exponential'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return exp(induced_local_field)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return neuron_output

#----------------------------------------------------------------- reciprocal ------------------------------------
cdef class ReciprocalActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'reciprocal'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return 1.0 / induced_local_field

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return - neuron_output ** 2

#----------------------------------------------------------------- square ------------------------------------
cdef class SquareActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'square'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return induced_local_field ** 2

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 2 * induced_local_field

#----------------------------------------------------------------- gauss ------------------------------------
cdef class GaussActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'gauss'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return exp(-induced_local_field ** 2)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return -2 * induced_local_field * neuron_output

#----------------------------------------------------------------- elliot ------------------------------------
cdef class ElliotActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'elliot'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return induced_local_field / (1 + fabs(induced_local_field))

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        aux = fabs(induced_local_field) + 1.0
        return (aux + induced_local_field) / (aux ** 2)

#----------------------------------------------------------------- sine ------------------------------------
cdef class SineActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'sine'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return sin(induced_local_field)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return cos(induced_local_field)

#----------------------------------------------------------------- cosine ------------------------------------
cdef class CosineActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'cosine'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return cos(induced_local_field)

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return -sin(induced_local_field)

#----------------------------------------------------------------- arctan ------------------------------------
cdef class ArctanActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'arctan'

    cpdef RealNumber original(self, RealNumber induced_local_field):
        return 2.0 * atan(induced_local_field) / pi

    cpdef RealNumber derivative(self, RealNumber induced_local_field, RealNumber neuron_output):
        return 2.0 / ((1 + induced_local_field ** 2) * pi)



##################################################################

activation_functions_set = {'tanh': TanhActivationFunction(),
                            'identity': IdentityActivationFunction(),
                            'threshold': ThresholdActivationFunction(),
                            'logistic': LogisticActivationFunction(),
                            'exponential': ExponentialActivationFunction(),
                            'reciprocal': ReciprocalActivationFunction(),
                            'square': SquareActivationFunction(),
                            'gauss': GaussActivationFunction(),
                            'sine': SineActivationFunction(),
                            'cosine': CosineActivationFunction(),
                            'elliot': ElliotActivationFunction(),
                            'arctan': ArctanActivationFunction(),
                            'default': ActivationFunction(),
                            }
