from pyAmore.cython.materials import *
from pyAmore.cython.network_predict_strategies import *
from pyAmore.cython.neuron_predict_strategies import *
from pyAmore.cython.network_fit_strategies import *
from pyAmore.cython.neuron_fit_strategies import *
from pyAmore.cython.builders import *
from pyAmore.cython.activation_functions import activation_functions_set


class Factory(metaclass=ABCMeta):
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

    def make_network_builder(self):
        return self.make('NetworkBuilder')

    def make_primitive_network(self):
        return self.make('Network', self)

    def make_primitive_container(self):
        return self.make('Container')

    def make_primitive_neuron(self, neural_network):
        return self.make('Neuron', neural_network)

    def make_primitive_connection(self, neuron):
        return self.make('Connection', neuron)

    def make_network_predict_strategy(self, neural_network):
        return self.make('NetworkPredictStrategy', neural_network)

    def make_network_fit_strategy(self, neural_network):
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
        last_layer = neuron.neural_network.layers[-1]
        if neuron in last_layer:
            return self.make('OutputNeuronFitStrategy', neuron)
        else:
            return self.make('HiddenNeuronFitStrategy', neuron)


class AdaptiveMaterialsFactory(MlpMaterialsFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    @abstractmethod
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({'NetworkFitStrategy': 'AdaptiveNetworkFitStrategy'})


class BatchMaterialsFactory(MlpMaterialsFactory):
    """ Simple implementation of a factory of multilayer feed forward network's elements
    """

    @abstractmethod
    def __init__(self):
        MlpMaterialsFactory.__init__(self)
        self.class_names.update({'NetworkFitStrategy': 'BatchNetworkFitStrategy'})


class AdaptiveGradientDescentMaterialsFactory(AdaptiveMaterialsFactory):
    def __init__(self):
        AdaptiveMaterialsFactory.__init__(self)
        self.class_names.update({name: 'AdaptiveGradientDescent' + name for name in (
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})


class AdaptiveGradientDescentWithMomentumMaterialsFactory(AdaptiveMaterialsFactory):
    def __init__(self):
        AdaptiveMaterialsFactory.__init__(self)
        self.class_names.update({name: 'AdaptiveGradientDescentWithMomentum' + name for name in (
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})


class BatchGradientDescentMaterialsFactory(BatchMaterialsFactory):
    def __init__(self):
        BatchMaterialsFactory.__init__(self)
        self.class_names.update({name: 'BatchGradientDescent' + name for name in (
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy',)})


class BatchGradientDescentWithMomentumMaterialsFactory(BatchMaterialsFactory):
    def __init__(self):
        BatchMaterialsFactory.__init__(self)
        self.class_names.update({name: 'BatchGradientDescentWithMomentum' + name for name in (
            'OutputNeuronFitStrategy',
            'HiddenNeuronFitStrategy')})
