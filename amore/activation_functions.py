from math import tanh, exp, cos, sin, fabs, atan, pi


##################################################################

def __default(induced_local_field: float, neuron_output: float):
    return tanh(induced_local_field)


def __default_derivative(induced_local_field, neuron_output):
    return 1 - neuron_output ** 2


__default.derivative = __default_derivative


##################################################################

def __tanh(induced_local_field: float, neuron_output: float):
    return tanh(induced_local_field)


def __tanh_derivative(induced_local_field, neuron_output):
    return 1.0 - neuron_output ** 2


__tanh.derivative = __tanh_derivative


##################################################################

def __identity(induced_local_field: float, neuron_output: float):
    return induced_local_field


def __identity_derivative(induced_local_field, neuron_output):
    return 1.0


__identity.derivative = __identity_derivative


##################################################################

def __threshold(induced_local_field: float, neuron_output: float):
    return 1.0 if induced_local_field > 0.0 else 0.0


def __threshold_derivative(induced_local_field, neuron_output):
    return 0.0


__threshold.derivative = __threshold_derivative


##################################################################

def __logistic(induced_local_field: float, neuron_output: float):
    return 1.0 / (1.0 + exp(-induced_local_field))


def __logistic_derivative(induced_local_field, neuron_output):
    return neuron_output * (1 - neuron_output)


__logistic.derivative = __logistic_derivative


##################################################################

def __exponential(induced_local_field: float, neuron_output: float):
    return exp(induced_local_field)


def __exponential_derivative(induced_local_field, neuron_output):
    return neuron_output


__exponential.derivative = __exponential_derivative


##################################################################

def __reciprocal(induced_local_field: float, neuron_output: float):
    return 1.0 / induced_local_field


def __reciprocal_derivative(induced_local_field, neuron_output):
    return - neuron_output ** 2


__reciprocal.derivative = __reciprocal_derivative


##################################################################

def __square(induced_local_field: float, neuron_output: float):
    return induced_local_field ** 2


def __square_derivative(induced_local_field, neuron_output):
    return 2 * induced_local_field


__square.derivative = __square_derivative


##################################################################

def __gauss(induced_local_field: float, neuron_output: float):
    return exp(-induced_local_field ** 2)


def __gauss_derivative(induced_local_field, neuron_output):
    return -2 * induced_local_field * neuron_output


__gauss.derivative = __gauss_derivative


##################################################################

def __sine(induced_local_field: float, neuron_output: float):
    return sin(induced_local_field)


def __sine_derivative(induced_local_field, neuron_output):
    return cos(induced_local_field)


__sine.derivative = __sine_derivative


##################################################################

def __cosine(induced_local_field: float, neuron_output: float):
    return cos(induced_local_field)


def __cosine_derivative(induced_local_field, neuron_output):
    return -sin(induced_local_field)


__cosine.derivative = __cosine_derivative


##################################################################

def __elliot(induced_local_field: float, neuron_output: float):
    return induced_local_field / (1 + fabs(induced_local_field))


def __elliot_derivative(induced_local_field, neuron_output):
    aux = fabs(induced_local_field) + 1.0
    return (aux + induced_local_field) / (aux ** 2)


__elliot.derivative = __elliot_derivative


##################################################################

def __arctan(induced_local_field: float, neuron_output: float):
    return 2.0 * atan(induced_local_field) / pi


def __arctan_derivative(induced_local_field, neuron_output):
    return 2.0 / ((1 + induced_local_field ** 2) * pi)


__arctan.derivative = __arctan_derivative

##################################################################

activation_functions_set = {'tanh': __tanh,
                            'identity': __identity,
                            'threshold': __threshold,
                            'logistic': __logistic,
                            'exponential': __exponential,
                            'reciprocal': __reciprocal,
                            'square': __square,
                            'gauss': __gauss,
                            'sine': __sine,
                            'cosine': __cosine,
                            'elliot': __elliot,
                            'arctan': __arctan,
                            'default': __default,
                            }
