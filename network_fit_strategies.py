from network_elements import *


class NetworkFitStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MlpNetworkFitStrategy(NetworkFitStrategy):
    @abstractmethod
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def write_targets_in_output_layer(self, data):
        for neuron, value in zip(self.neural_network.layers[-1], data):
            neuron.target = value

    def single_pattern_backward_action(self):
        for layer in reversed(self.neural_network.layers[1:]):
            for neuron in layer:
                neuron.fit()

    def set_neurons_fit_strategy(self, neural_factory):
        # Hidden Layers
        for layer in self.neural_network.layers[1:-1]:
            for neuron in layer:
                neuron.fit_strategy = neural_factory.make_hidden_neuron_fit_strategy(neuron)
        # Output Layers
        for neuron in self.neural_network.layers[-1]:
            neuron.fit_strategy = neural_factory.make_output_neuron_fit_strategy(neuron)

    def set_neurons_learning_rate(self, learning_rate):
        for layer in self.neural_network.layers[1:]:
            for neuron in layer:
                neuron.learning_rate = learning_rate

    def set_learning_rate(self, learning_rate):
        self.neural_network.set_learning_rate(learning_rate)


class AdaptiveGradientDescentNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def __call__(self, *args, **kwargs):
        self.set_learning_rate(learning_rate)
        max_shows = number_of_epochs / show_step if number_of_epochs > show_step else 1

        for dummy_1 in range(max_shows):
            for dummy_2 in range(show_step):
                for input_data_row, target_data_row in zip(input_data, target_data):
                    self.neural_network.read(input_data_row)
                    self.neural_network.activate_neurons()
                    self.neural_network.write_target(target_data_row)
                    self.neural_network.single_pattern_backward_action()
