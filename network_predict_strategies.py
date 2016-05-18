from neuron_fit_strategy import *
import numpy


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

    def __call__(self, input_data):
        data_number_of_rows, data_number_of_columns = input_data.shape
        input_layer_size, *hidden_layers_size, output_layer_size = self.neural_network.shape

        if input_layer_size != data_number_of_columns:
            raise ValueError(
                '\n[SimpleNeuralNetwork.sim Error:] Input layer size different from data number of columns\n')

        output_data = numpy.zeros((data_number_of_rows, output_layer_size))
        for row, input_data in enumerate(input_data):
            self.neural_network.read(input_data)
            self.activate_neurons()
            output_data[row, :] = self.neural_network.inspect_output()
        return output_data

    def activate_neurons(self):
        for layer in self.neural_network.layers[1:]:
            for neuron in layer:
                neuron()


class AdaptiveGradientDescentNetworkPredictStrategy(MlpNetworkPredictStrategy):
    def __init__(self, neural_network):
        MlpNetworkPredictStrategy.__init__(self, neural_network)

# class AdaptiveGradientDescentWithMomentumNetworkPredictStrategy(MlpNetworkPredictStrategy):
#     def __init__(self, neural_network):
#         MlpNetworkPredictStrategy.__init__(self, neural_network)
#
#
# class BatchGradientDescentNetworkPredictStrategy(MlpNetworkPredictStrategy):
#     def __init__(self, neural_network):
#         MlpNetworkPredictStrategy.__init__(self, neural_network)
#
#
# class BatchGradientDescentWithMomentumNetworkPredictStrategy(MlpNetworkPredictStrategy):
#     def __init__(self, neural_network):
#         MlpNetworkPredictStrategy.__init__(self, neural_network)
