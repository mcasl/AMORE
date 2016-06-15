from common cimport RealNumber

#from network_fit_strategies     cimport *
#from network_predict_strategies cimport *
#from neuron_fit_strategies      cimport *
#from neuron_predict_strategies  cimport *


cdef class MlpContainer(list):
    pass

cdef class Network(object):
    cdef public:
        object factory
        #NetworkPredictStrategy predict_strategy
        object predict_strategy

        #NetworkFitStrategy fit_strategy
        object fit_strategy

cdef class MlpNetwork(Network):
    cdef public:
        MlpContainer layers

cdef class Neuron(object):
    cdef public:
        object label
        RealNumber output
        # Network neural_network
        object neural_network
        #        NeuronPredictStrategy predict_strategy
        #        NeuronFitStrategy fit_strategy
        object predict_strategy
        object fit_strategy

cdef class MlpNeuron(Neuron):
    cdef public:
        object activation_function  # TODO: must be change to specific type
        MlpContainer connections
        RealNumber bias

cdef class MlpConnection(object):
    cdef public:
        RealNumber weight
        Neuron neuron
