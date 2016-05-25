from .activation_functions import activation_functions_set
from .cost_functions import cost_functions_set
from .builders import *
from .materials import *
from .network_fit_strategies import *
from .network_predict_strategies import *
from .neuron_fit_strategies import *
from .neuron_predict_strategies import *


class Factory(object, metaclass=ABCMeta):
    """ The mother of all neural factories (a.k.a Interface)
    """
    version = 1.1

    @abstractmethod
    def __init__(self):
        self.class_names = {}

    def make(self, class_wanted, *pargs, **kwargs):
        class_complete_name = self.class_names[class_wanted]
        return globals()[class_complete_name](*pargs, **kwargs)

    @abstractmethod
    def make_neural_network_builder(self):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_builder")

    @abstractmethod
    def make_neural_network_predict_strategy(self, neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_predict_strategy")

    @abstractmethod
    def make_neural_network_fit_strategy(self, neural_network):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neural_network_fit_strategy")

    @abstractmethod
    def make_neuron_predict_strategy(self, neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_predict_strategy")

    @abstractmethod
    def make_neuron_fit_strategy(self, neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_fit_strategy")


class MlpFactory(Factory):
    """ Simple implementation of a factories of multilayer feed forward network's elements
    """
    @abstractmethod
    def __init__(self):
        Factory.__init__(self)
        self.class_names = {}

    def make_primitive_connection(self, neuron):
        return self.make('connection', neuron)

    def make_primitive_container(self):
        return []

    def make_primitive_neuron(self, neural_network):
        return self.make('neuron', neural_network)

    def make_primitive_neural_network(self):
        return self.make('neural_network', self)

    def make_neural_network_builder(self):
        return self.make('neural_network_builder')

    def make_neuron_predict_strategy(self, neuron):
        return self.make('neuron_predict_strategy', neuron)

    def make_neuron_fit_strategy(self, neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return self.make('output_neuron_fit_strategy', neuron)
        else:
            return self.make('hidden_neuron_fit_strategy', neuron)

    def make_neural_network_predict_strategy(self, neural_network):
        return self.make('neural_network_predict_strategy', neural_network)

    def make_neural_network_fit_strategy(self, neural_network):
        return self.make('neural_network_fit_strategy', neural_network)


    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]


class AdaptiveGradientDescentFactory(MlpFactory):
    def __init__(self):
        MlpFactory.__init__(self)

        self.class_names = {
            'connection': 'Connection',
            'container': 'list',
            'neuron': 'MlpNeuron',
            'neural_network': 'MlpNeuralNetwork',
            'neural_network_builder': 'MlpNeuralNetworkBuilder',
            'neural_network_fit_strategy': 'AdaptiveGradientDescentNetworkFitStrategy',
            'neural_network_predict_strategy': 'AdaptiveGradientDescentNetworkPredictStrategy',
            'neuron_fit_strategy': 'AdaptiveGradientDescentNeuronFitStrategy',
            'output_neuron_fit_strategy': 'AdaptiveGradientDescentOutputNeuronFitStrategy',
            'hidden_neuron_fit_strategy': 'AdaptiveGradientDescentHiddenNeuronFitStrategy',
            'neuron_predict_strategy': 'AdaptiveGradientDescentNeuronPredictStrategy',
        }


class AdaptiveGradientDescentWithMomentumFactory(MlpFactory):
    def __init__(self):
        MlpFactory.__init__(self)
        self.class_names = {
            'connection': 'Connection',
            'container': 'list',
            'neuron': 'MlpNeuron',
            'neural_network': 'MlpNeuralNetwork',
            'neural_network_builder': 'MlpNeuralNetworkBuilder',
            'neural_network_fit_strategy': 'AdaptiveGradientDescentWithMomentumNetworkFitStrategy',
            'neural_network_predict_strategy': 'AdaptiveGradientDescentWithMomentumNetworkPredictStrategy',
            'neuron_fit_strategy': 'AdaptiveGradientDescentWithMomentumNeuronFitStrategy',
            'output_neuron_fit_strategy': 'AdaptiveGradientDescentWithMomentumOutputNeuronFitStrategy',
            'hidden_neuron_fit_strategy': 'AdaptiveGradientDescentWithMomentumHiddenNeuronFitStrategy',
            'neuron_predict_strategy': 'AdaptiveGradientDescentWithMomentumNeuronPredictStrategy',
        }


class BatchGradientDescentFactory(MlpFactory):
    def __init__(self):
        MlpFactory.__init__(self)
        self.class_names = {
            'connection': 'Connection',
            'container': 'list',
            'neuron': 'MlpNeuron',
            'neural_network': 'MlpNeuralNetwork',
            'neural_network_builder': 'MlpNeuralNetworkBuilder',
            'neural_network_fit_strategy': 'BatchGradientDescentNetworkFitStrategy',
            'neural_network_predict_strategy': 'BatchGradientDescentNetworkPredictStrategy',
            'neuron_fit_strategy': 'BatchGradientDescentNeuronFitStrategy',
            'output_neuron_fit_strategy': 'BatchGradientDescentOutputNeuronFitStrategy',
            'hidden_neuron_fit_strategy': 'BatchGradientDescentHiddenNeuronFitStrategy',
            'neuron_predict_strategy': 'BatchGradientDescentPredictStrategy',
        }


class BatchGradientDescentWithMomentumFactory(MlpFactory):
    def __init__(self):
        MlpFactory.__init__(self)
        self.class_names = {
            'connection': 'Connection',
            'container': 'list',
            'neuron': 'MlpNeuron',
            'neural_network': 'MlpNeuralNetwork',
            'neural_network_builder': 'MlpNeuralNetworkBuilder',
            'neural_network_fit_strategy': 'BatchGradientDescentWithMomentumNetworkFitStrategy',
            'neural_network_predict_strategy': 'BatchGradientDescentWithMomentumNetworkPredictStrategy',
            'neuron_fit_strategy': 'BatchGradientDescentWithMomentumNeuronFitStrategy',
            'output_neuron_fit_strategy': 'BatchGradientDescentWithMomentumOutputNeuronFitStrategy',
            'hidden_neuron_fit_strategy': 'BatchGradientDescentWithMomentumHiddenNeuronFitStrategy',
            'neuron_predict_strategy': 'BatchGradientDescentWithMomentumNeuronPredictStrategy',
        }
