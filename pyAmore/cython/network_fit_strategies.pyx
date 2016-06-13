from pyAmore.cython.neuron_predict_strategies import *

cdef class NetworkFitStrategy(object):
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NetworkFitStrategy.__call__")

cdef class MlpNetworkFitStrategy(NetworkFitStrategy):
    def __init__(self, neural_network):
        NetworkFitStrategy.__init__(self, neural_network)
        self.neuron_fit_sequence = neural_network.factory.make_primitive_container()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling MlpNetworkFitStrategy.__call__")

    def poke_targets(self, data):
        output_layer = self.neural_network.layers[-1]
        for neuron, value in zip(output_layer, data):
            neuron.fit_strategy.target = value

    cpdef backpropagate(self):
        cdef int position
        cdef neuron_fit_sequence_length = len(self.neuron_fit_sequence)
        for position in range(neuron_fit_sequence_length):
            neuron = self.neuron_fit_sequence[position]
            neuron.fit_strategy()

cdef class AdaptiveNetworkFitStrategy(MlpNetworkFitStrategy):
    def __init__(self, neural_network):
        MlpNetworkFitStrategy.__init__(self, neural_network)

    def __call__(self, input_data, target_data):
        for input_data_row, target_data_row in zip(input_data, target_data):
            self.neural_network.poke_inputs(input_data_row)
            self.neural_network.predict_strategy.activate_neurons()
            self.neural_network.fit_strategy.poke_targets(target_data_row)
            self.neural_network.fit_strategy.backpropagate()
