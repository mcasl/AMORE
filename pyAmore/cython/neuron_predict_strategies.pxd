
cdef class NeuronPredictStrategy:
    cdef Neuron neuron

cdef class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    cdef double induced_local_field


