import numpy as np

from .neuron_predict_strategies import *


class NetworkPredictStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network):
        self.neural_network = neural_network

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.__call__")

    @abstractmethod
    def activate_neurons(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkPredictStrategy.activate_neurons")


class MlpNetworkPredictStrategy(NetworkPredictStrategy):
    def __init__(self, neural_network):
        NetworkPredictStrategy.__init__(self, neural_network)
        self.neuron_predict_sequence = []

    def __call__(self, input_data):
        data_number_of_rows, data_number_of_columns = input_data.shape
        input_layer_size, *hidden_layers_size, output_layer_size = self.neural_network.shape

        if input_layer_size != data_number_of_columns:
            raise ValueError(
                '\n[SimpleNeuralNetwork.sim Error:] Input layer size different from data number of columns\n')

        output_data = np.zeros((data_number_of_rows, output_layer_size))
        for row, input_data in enumerate(input_data):
            self.neural_network.poke_inputs(input_data)
            self.activate_neurons()
            output_data[row, :] = self.neural_network.pick_outputs()
        return output_data

    def activate_neurons(self):
        for neuron in self.neuron_predict_sequence:
            neuron.predict_strategy()

