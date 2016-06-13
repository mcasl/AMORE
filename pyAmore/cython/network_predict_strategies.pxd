from materials cimport *

cdef class NetworkPredictStrategy(object):
    cdef public Network neural_network

cdef class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    cdef public MlpContainer neuron_predict_sequence
    cpdef activate_neurons(MlpNetworkPredictStrategy self)
