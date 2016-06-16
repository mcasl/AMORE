from abc import ABCMeta, abstractmethod


class NeuronPredictStrategy(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        self.neuron = neuron

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NeuronPredictStrategy.predict")


class MlpNeuronPredictStrategy(NeuronPredictStrategy):
    def __init__(self, neuron):
        NeuronPredictStrategy.__init__(self, neuron)
        self.induced_local_field = 0.0

    def predict(self):
        accumulator = self.neuron.bias
        for connection in self.neuron.connections:
            accumulator += connection.neuron.output * connection.weight
        self.induced_local_field = accumulator

        self.neuron.output = self.neuron.activation_function.original(self.induced_local_field)
        return self.neuron.output
