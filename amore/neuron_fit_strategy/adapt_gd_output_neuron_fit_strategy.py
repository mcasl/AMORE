from .adapt_gd_neuron_fit_strategy import AdaptiveGradientDescentNeuronFitStrategy


class AdaptiveGradientDescentOutputNeuronFitStrategy(AdaptiveGradientDescentNeuronFitStrategy):
    def __init__(self, neuron):
        AdaptiveGradientDescentNeuronFitStrategy.__init__(self, neuron)

    def __call__(self):
        neuron = self.neuron
        self.output_derivative = neuron.activation_function.derivative(neuron.predict_strategy.induced_local_field)
        self.delta = self.output_derivative * self.cost_function.derivative(neuron.output, self.target)
        minus_learning_rate_x_delta = -self.learning_rate * self.delta
        neuron.bias += minus_learning_rate_x_delta
        for connection in self.neuron.connections:
            input_value = connection.neuron.output
            connection.weight += minus_learning_rate_x_delta * input_value
            connection.neuron.fit_strategy.delta += self.delta * connection.weight
        self.delta = 0.0
