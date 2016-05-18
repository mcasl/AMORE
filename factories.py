from neural_network_builders import *
from neuron_predict_strategies import *
from container import *
from activation_functions import *
from cost_functions import *


class NeuralFactory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """

    @staticmethod
    @abstractmethod
    def make_neural_network_builder():
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        pass

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        pass


class MlpFactory(NeuralFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    @staticmethod
    def make_primitive_connection(neuron):
        return Connection(neuron)

    @staticmethod
    def make_primitive_container():
        return Container()

    @staticmethod
    def make_primitive_neuron(neural_network):
        neuron = MlpNeuron(neural_network)
        return neuron

    def make_primitive_layer(self, size, neural_network):
        layer = self.make_primitive_container()
        for dummy in range(size):
            layer.append(self.make_primitive_neuron(neural_network))
        return layer

    def make_primitive_neural_network(self):
        neural_network = MlpNeuralNetwork(self)
        neural_network.predict_strategy = self.make_neural_network_predict_strategy(neural_network)
        neural_network.layers = self.make_primitive_container()
        return neural_network

    @staticmethod
    def make_neural_network_builder():
        return MlpNeuralNetworkBuilder()

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neural_network):
        pass

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        pass

    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]


class AdaptiveGradientDescentFactory(MlpFactory):
    @staticmethod
    def make_neuron_predict_strategy(neuron):
        return AdaptiveGradientDescentNeuronPredictStrategy(neuron)

    @staticmethod
    def make_neuron_fit_strategy(neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return AdaptiveGradientDescentOutputNeuronFitStrategy(neuron)
        else:
            return AdaptiveGradientDescentHiddenNeuronFitStrategy(neuron)

    @staticmethod
    def make_neural_network_predict_strategy(neural_network):
        return AdaptiveGradientDescentNetworkPredictStrategy(neural_network)

    @staticmethod
    def make_neural_network_fit_strategy(neural_network):
        return AdaptiveGradientDescentNetworkFitStrategy(neural_network)

#
# class AdaptiveGradientDescentWithMomentumFactory(MlpFactory):
#
#     def make_neuron_fit_strategy(self, neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return AdaptiveGradientDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return AdaptiveGradientDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return AdaptiveGradientDescentWithMomentumNetworkPredictStrategy(neural_network)
#
#
# class BatchGradientDescentFactory(MlpFactory):
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchGradientDescentOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchGradientDescentHiddenNeuronFitStrategy(neuron)
#
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return BatchGradientDescentNetworkPredictStrategy(neural_network)
#
#
# class BatchGradientDescentWithMomentumFactory(MlpFactory):
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#
#     def make_neural_network_predict_strategy(self, neural_network):
#         return BatchDescentWithMomentumNetworkPredictStrategy(neural_network)
