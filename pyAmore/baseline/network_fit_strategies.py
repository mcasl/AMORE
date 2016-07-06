from .neuron_predict_strategies import *


class NetworkFitStrategy(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkFitStrategy.predict")


class MlpNetworkFitStrategy(NetworkFitStrategy):
    @abstractmethod
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling MlpNetworkFitStrategy.predict")

    def poke_targets(self, data):
        output_layer = self.neural_network.layers[-1]
        for neuron, value in zip(output_layer, data):
            neuron.fit_strategy.target = value

    def backpropagate(self):
        for layer in reversed(self.neural_network.layers):
            for neuron in layer:
                neuron.fit_strategy.fit()


class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def fit(self, input_data, target_data):
        for input_data_row, target_data_row in zip(input_data, target_data):
            self.neural_network.poke_inputs(input_data_row)
            self.neural_network.predict_strategy.activate_neurons()
            self.neural_network.fit_strategy.poke_targets(target_data_row)
            self.neural_network.fit_strategy.backpropagate()
