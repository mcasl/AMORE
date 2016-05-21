from .builders import *
from .cost_functions import *
from .neuron_predict_strategies import *


class NeuronFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        self.neuron = neuron
        self.cost_function = cost_functions_set['adaptLMS']


class MlpNeuronFitStrategy(NeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        NeuronFitStrategy.__init__(self, neuron)


class AdaptiveGradientDescentNeuronFitStrategy(MlpNeuronFitStrategy):
    @abstractmethod
    def __init__(self, neuron):
        MlpNeuronFitStrategy.__init__(self, neuron)
        self.delta = 0.0
        self.learning_rate = 0.1
        self.output_derivative = 0.0
        self.target = 0.0


class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    def __call__(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field)
        self.delta = self.output_derivative * self.cost_function.derivative(neuron.output, self.target)
        minus_learning_rate_x_delta = -self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0


class AdaptiveGradientDescentHiddenNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    def __call__(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field)
        self.delta *= self.output_derivative
        minus_learning_rate_x_delta = - self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0
