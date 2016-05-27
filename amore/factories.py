from .activation_functions import activation_functions_set
from .cost_functions import cost_functions_set
from .builders import *
from .materials import *
from .network_fit_strategies import *
from .network_predict_strategies import *
from .neuron_fit_strategies import *
from .neuron_predict_strategies import *


class Factory(object, metaclass=ABCMeta):
    """ An class for providing a generic make method
    """

    def __init__(self):
        self.class_names = {}

    def make(self, key, *pargs, **kwargs):
        class_complete_name = self.class_names[key]
        return globals()[class_complete_name](*pargs, **kwargs)


class MaterialsFactory(Factory):
    """ The mother of all neural factories. It provides code that is common for all subclasses. Each of these
        will use it in accordance to its own self.class_names dictionary
    """

    @abstractmethod
    def __init__(self):
        Factory.__init__(self)

    def make_neural_network_builder(self):
        return self.make('NetworkBuilder')

    def make_primitive_neural_network(self):
        return self.make('Network', self)

    def make_primitive_container(self):
        return self.make('Container')

    def make_primitive_neuron(self, neural_network):
        return self.make('Neuron', neural_network)

    def make_primitive_connection(self, neuron):
        return self.make('Connection', neuron)

    def make_neural_network_predict_strategy(self, neural_network):
        return self.make('NetworkPredictStrategy', neural_network)

    def make_neural_network_fit_strategy(self, neural_network):
        return self.make('NetworkFitStrategy', neural_network)

    def make_neuron_predict_strategy(self, neuron):
        return self.make('NeuronPredictStrategy', neuron)

    @abstractmethod
    def make_neuron_fit_strategy(self, neuron):
        raise NotImplementedError("You shouldn't be calling NeuralFactory.make_neuron_fit_strategy")

    @staticmethod
    def make_activation_function(function_name):
        return activation_functions_set[function_name]

    @staticmethod
    def make_cost_function(function_name):
        return cost_functions_set[function_name]


class MlpMaterialsFactory(MaterialsFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    @abstractmethod
    def __init__(self):
        MaterialsFactory.__init__(self)
        self.class_names.update({name: 'Mlp' + name for name in (
            'Connection',
            'Container',
            'Neuron',
            'Network',
            'NeuronPredictStrategy',
            'NetworkPredictStrategy',
            'NetworkBuilder')})


    def make_neuron_fit_strategy(self, neuron):
        is_output_neuron = neuron in neuron.neural_network.layers[-1]
        if is_output_neuron:
            return self.make('OutputNeuronFitStrategy', neuron)
        else:
            return self.make('HiddenNeuronFitStrategy', neuron)


class AdaptiveGradientDescentFactory(MlpMaterialsFactory):
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({name: 'AdaptiveGradientDescent' + name for name in (
            'NetworkFitStrategy',
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})


class AdaptiveGradientDescentWithMomentumFactory(MlpMaterialsFactory):
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({name: 'AdaptiveGradientDescentWithMomentum' + name for name in (
            'NetworkFitStrategy',
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})


class BatchGradientDescentFactory(MlpMaterialsFactory):
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({name: 'BatchGradientDescent' + name for name in (
            'NetworkFitStrategy',
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy',)})


class BatchGradientDescentWithMomentumFactory(MlpMaterialsFactory):
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({name: 'BatchGradientDescentWithMomentum' + name for name in (
            'NetworkFitStrategy',
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})
