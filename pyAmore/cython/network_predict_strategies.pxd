from materials cimport Network, MlpContainer

cdef class NetworkPredictStrategy:
    cdef public Network neural_network

cdef class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    cdef public MlpContainer neuron_predict_sequence
    cpdef activate_neurons(self)
