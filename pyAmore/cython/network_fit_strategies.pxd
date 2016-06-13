from materials cimport *

cdef class NetworkFitStrategy(object):
    cdef public Network neural_network

cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    cdef public MlpContainer neuron_fit_sequence
    cpdef backpropagate(MlpNetworkFitStrategy self)

cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    pass
