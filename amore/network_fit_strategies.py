from .neuron_predict_strategies import *


class NetworkFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkFitStrategy.__call__")


class MlpNetworkFitStrategy(NetworkFitStrategy):
    @abstractmethod
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)
        self.neuron_fit_sequence = []

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling MlpNetworkFitStrategy.__call__")

    def write_targets_in_output_layer(self, data):
        for neuron, value in zip(self.neural_network.layers[-1], data):
            neuron.fit_strategy.target = value

    def single_pattern_backward_action(self):
        [neuron.fit_strategy() for neuron in self.neuron_fit_sequence]

    def set_neurons_fit_strategy(self, neural_factory):
        # Providing input neurons with a fit strategy simplifies
        # the code for adjusting the weights and biases
        for layer in self.neural_network.layers:
            for neuron in layer:
                neuron.fit_strategy = neural_factory.make_neuron_fit_strategy(neuron)

    def set_neurons_learning_rate(self, learning_rate):
        for layer in self.neural_network.layers:
            for neuron in layer:
                neuron.fit_strategy.learning_rate = learning_rate


class AdaptiveGradientDescentNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def __call__(self, input_data, target_data):
        for input_data_row, target_data_row in zip(input_data, target_data):
            self.neural_network.read(input_data_row)
            self.neural_network.predict_strategy.activate_neurons()
            self.neural_network.fit_strategy.write_targets_in_output_layer(target_data_row)
            self.neural_network.fit_strategy.single_pattern_backward_action()
