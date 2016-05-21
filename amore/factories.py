from .container import *
from .materials import *
from .network_fit_strategies import *
from .network_predict_strategies import *
from .neuron_fit_strategies import *
from .neuron_predict_strategies import *
from .activation_functions import activation_functions_set


class NeuralFactory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.__init__")

    @staticmethod
    @abstractmethod
    def make_neural_network_builder():
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_builder")

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_fit_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_predict_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_fit_strategy")


class MlpFactory(NeuralFactory):
    """ Simple implementation of a factories of multilayer feed forward network's elements
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("You shouldn't be calling MlpFactory.__init__")

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
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neuron_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neuron_fit_strategy(neuron):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neuron_fit_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_predict_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neural_network_predict_strategy")

    @staticmethod
    @abstractmethod
    def make_neural_network_fit_strategy(neural_network):
        raise NotImplementedError("You shouldn't be calling MlpFactory.make_neural_network_fit_strategy")

    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]


class AdaptiveGradientDescentFactory(MlpFactory):
    def __init__(self):
        pass

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

# from amore.neuron_fit_strategies.adapt_gdwm_neuron_fit_strategy import *
# from amore.factories.mlp_factory import MlpFactory

#
# class AdaptiveGradientDescentWithMomentumFactory(MlpFactory):
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def make_neuron_predict_strategy(neuron):
#         return AdaptiveGradientDescentWithMomentumNeuronPredictStrategy(neuron)
#
#     @staticmethod
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return AdaptiveGradientDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return AdaptiveGradientDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#     @staticmethod
#     def make_neural_network_predict_strategy(neural_network):
#         return AdaptiveGradientDescentWithMomentumNetworkPredictStrategy(neural_network)
#
#     @staticmethod
#     def make_neural_network_fit_strategy(neural_network):
#         return AdaptiveGradientDescentWithMomentumNetworkFitStrategy(neural_network)
# from .mlp_factory import MlpFactory

#
# class BatchGradientDescentFactory(MlpFactory):
#     @staticmethod
#     def make_neural_network_fit_strategy(neural_network):
#         pass
#
#     @staticmethod
#     def make_neuron_predict_strategy(neuron):
#         pass
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchGradientDescentOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchGradientDescentHiddenNeuronFitStrategy(neuron)
#
#     @staticmethod
#     def make_neural_network_predict_strategy(neural_network):
#         return BatchGradientDescentNetworkPredictStrategy(neural_network)
# from .mlp_factory import MlpFactory

#
# class BatchGradientDescentWithMomentumFactory(MlpFactory):
#     @staticmethod
#     def make_neural_network_fit_strategy(neural_network):
#         pass
#
#     @staticmethod
#     def make_neuron_predict_strategy(neuron):
#         pass
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def make_neuron_fit_strategy(neuron):
#         is_output_neuron = neuron in neuron.neural_network.layers[-1]
#         if is_output_neuron:
#             return BatchDescentWithMomentumOutputNeuronFitStrategy(neuron)
#         else:
#             return BatchDescentWithMomentumHiddenNeuronFitStrategy(neuron)
#
#     @staticmethod
#     def make_neural_network_predict_strategy(neural_network):
#         return BatchDescentWithMomentumNetworkPredictStrategy(neural_network)
