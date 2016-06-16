# cython: profile=True
from common cimport *
from materials cimport *

cdef class NeuronPredictStrategy:
    def __init__(self, Neuron neuron):
        self.neuron = neuron

    cpdef RealNumber predict(self):
        raise NotImplementedError("You shouldn't be calling NeuronPredictStrategy.predict")

cdef class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    def __init__(self, MlpNeuron neuron):
        NeuronPredictStrategy.__init__(self, neuron)
        self.induced_local_field = 0.0

    cpdef RealNumber predict(self):
        cdef MlpNeuron neuron = self.neuron
        cdef MlpContainer neuron_connections = neuron.connections
        cdef RealNumber accumulator = neuron.bias
        cdef int position
        cdef int self_neuron_connection_length = len(neuron_connections)
        cdef MlpConnection connection
        for position in range(self_neuron_connection_length):
            connection = neuron_connections[position]
            accumulator += connection.neuron.output * connection.weight
        self.induced_local_field = accumulator

        neuron.output = neuron.activation_function.original(self.induced_local_field)
        return neuron.output
