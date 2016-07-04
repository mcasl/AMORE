from materials cimport Network, MlpContainer


cdef class NetworkFitStrategy:
    cdef public:
        Network neural_network

cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    cpdef poke_targets(self, data)
    cpdef backpropagate(self)


cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    pass
