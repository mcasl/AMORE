from materials cimport Network, MlpContainer

cdef class NetworkPredictStrategy:
    cdef public Network neural_network
    cpdef activate_neurons(self)

cdef class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    cpdef activate_neurons(self)
