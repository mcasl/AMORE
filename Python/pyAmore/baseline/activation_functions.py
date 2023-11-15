from math import tanh, exp, cos, sin, fabs, atan, pi


class ActivationFunction:
    def __init__(self):
        self.label = 'default'

    def original(self, induced_local_field):
        return tanh(induced_local_field)

    def derivative(self, induced_local_field, neuron_output):
        return 1.0 - neuron_output ** 2


# ----------------------------------------------------------------- tanh ------------------------------------
class TanhActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'tanh'

    def original(self, induced_local_field):
        return tanh(induced_local_field)

    def derivative(self, induced_local_field, neuron_output):
        return 1.0 - neuron_output ** 2


# ----------------------------------------------------------------- identity ------------------------------------
class IdentityActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'identity'

    def original(self, induced_local_field):
        return induced_local_field

    def derivative(self, induced_local_field, neuron_output):
        return 1.0


# ----------------------------------------------------------------- threshold ------------------------------------
class ThresholdActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'threshold'

    def original(self, induced_local_field):
        return 1.0 if induced_local_field > 0.0 else 0.0

    def derivative(self, induced_local_field, neuron_output):
        return 0.0


# ----------------------------------------------------------------- logistic ------------------------------------
class LogisticActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'logistic'

    def original(self, induced_local_field):
        return 1.0 / (1.0 + exp(-induced_local_field))

    def derivative(self, induced_local_field, neuron_output):
        return neuron_output * (1 - neuron_output)


# ----------------------------------------------------------------- exponential ------------------------------------
class ExponentialActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'exponential'

    def original(self, induced_local_field):
        return exp(induced_local_field)

    def derivative(self, induced_local_field, neuron_output):
        return neuron_output


# ----------------------------------------------------------------- reciprocal ------------------------------------
class ReciprocalActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'reciprocal'

    def original(self, induced_local_field):
        return 1.0 / induced_local_field

    def derivative(self, induced_local_field, neuron_output):
        return - neuron_output ** 2


# ----------------------------------------------------------------- square ------------------------------------
class SquareActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'square'

    def original(self, induced_local_field):
        return induced_local_field ** 2

    def derivative(self, induced_local_field, neuron_output):
        return 2 * induced_local_field


# ----------------------------------------------------------------- gauss ------------------------------------
class GaussActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'gauss'

    def original(self, induced_local_field):
        return exp(-induced_local_field ** 2)

    def derivative(self, induced_local_field, neuron_output):
        return -2 * induced_local_field * neuron_output


# ----------------------------------------------------------------- elliot ------------------------------------
class ElliotActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'elliot'

    def original(self, induced_local_field):
        return induced_local_field / (1 + fabs(induced_local_field))

    def derivative(self, induced_local_field, neuron_output):
        aux = fabs(induced_local_field) + 1.0
        return (aux + induced_local_field) / (aux ** 2)


# ----------------------------------------------------------------- sine ------------------------------------
class SineActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'sine'

    def original(self, induced_local_field):
        return sin(induced_local_field)

    def derivative(self, induced_local_field, neuron_output):
        return cos(induced_local_field)


# ----------------------------------------------------------------- cosine ------------------------------------
class CosineActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'cosine'

    def original(self, induced_local_field):
        return cos(induced_local_field)

    def derivative(self, induced_local_field, neuron_output):
        return -sin(induced_local_field)


# ----------------------------------------------------------------- arctan ------------------------------------
class ArctanActivationFunction(ActivationFunction):
    def __init__(self):
        self.label = 'arctan'

    def original(self, induced_local_field):
        return 2.0 * atan(induced_local_field) / pi

    def derivative(self, induced_local_field, neuron_output):
        return 2.0 / ((1 + induced_local_field ** 2) * pi)



##################################################################

activation_functions_set = {'tanh': TanhActivationFunction(),
                            'identity': IdentityActivationFunction(),
                            'threshold': ThresholdActivationFunction(),
                            'logistic': LogisticActivationFunction(),
                            'exponential': ExponentialActivationFunction(),
                            'reciprocal': ReciprocalActivationFunction(),
                            'square': SquareActivationFunction(),
                            'gauss': GaussActivationFunction(),
                            'sine': SineActivationFunction(),
                            'cosine': CosineActivationFunction(),
                            'elliot': ElliotActivationFunction(),
                            'arctan': ArctanActivationFunction(),
                            'default': ActivationFunction(),
                            }
