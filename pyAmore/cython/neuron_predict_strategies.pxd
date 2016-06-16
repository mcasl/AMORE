from common cimport *
from materials cimport *

cdef class NeuronPredictStrategy:
    cdef public:
        Neuron neuron
    cpdef RealNumber predict(self)

cdef class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    cdef public:
        RealNumber induced_local_field
    cpdef RealNumber predict(self)
