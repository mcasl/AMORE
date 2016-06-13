from materials cimport *

cdef class NetworkFitStrategy:
    cdef public Network neural_network

cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    cdef public MlpContainer neuron_fit_sequence
    cpdef poke_targets(MlpNetworkFitStrategy self, RealNumber[:] data)
    cpdef backpropagate(MlpNetworkFitStrategy self)

cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    pass
