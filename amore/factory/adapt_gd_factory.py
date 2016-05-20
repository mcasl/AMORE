from .mlp_factory import MlpFactory


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
