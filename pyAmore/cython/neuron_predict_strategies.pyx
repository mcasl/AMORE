cdef class NeuronPredictStrategy(object):
    def __init__(self, neuron):
        self.neuron = neuron

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NeuronPredictStrategy.__call__")


cdef class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    def __init__(self, neuron):
        NeuronPredictStrategy.__init__(self, neuron)
        self.induced_local_field = 0.0

    def __call__(self):
        accumulator = self.neuron.bias
        for connection in self.neuron.connections:
            accumulator += connection.neuron.output * connection.weight
        self.induced_local_field = accumulator

        self.neuron.output = self.neuron.activation_function(self.induced_local_field)
        return self.neuron.output
