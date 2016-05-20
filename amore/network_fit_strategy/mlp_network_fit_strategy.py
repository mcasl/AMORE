from abc import abstractmethod
from .networkfitstrategy import NetworkFitStrategy


class MlpNetworkFitStrategy(NetworkFitStrategy):
    @abstractmethod
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling MlpNetworkFitStrategy.__call__")

    def write_targets_in_output_layer(self, data):
        for neuron, value in zip(self.neural_network.layers[-1], data):
            neuron.fit_strategy.target = value

    def single_pattern_backward_action(self):
        for layer in reversed(self.neural_network.layers[1:]):
            for neuron in layer:
                neuron.fit_strategy()

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

