from common cimport *
from materials cimport *

cdef class NeuronPredictStrategy(object):
    cdef public Neuron neuron

cdef class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    cdef public RealNumber induced_local_field
