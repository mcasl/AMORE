from materials cimport Network, MlpContainer
cimport numpy as np

cdef class NetworkFitStrategy:
    cdef public:
        Network neural_network
    cpdef fit(self, np.ndarray input_data, np.ndarray target_data)

cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    cpdef poke_targets(self, data)
    cpdef backpropagate(self)

cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    pass
