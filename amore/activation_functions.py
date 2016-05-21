from math import tanh, exp, cos, sin, fabs, atan, pi


##################################################################

def __default(x: float):
    return tanh(x)


__default.derivative = lambda x: 1 - tanh(x) ** 2


##################################################################

def __tanh(x: float):
    return tanh(x)


__tanh.derivative = lambda x: 1 - tanh(x) ** 2


##################################################################

def __identity(x: float):
    return x


__identity.derivative = lambda x: 1


##################################################################

def __threshold(x: float):
    return 1 if x > 0 else 0


__threshold.derivative = lambda x: 0


##################################################################

def __logistic(x: float):
    return 1.0 / (1.0 + exp(-x))


def __logistic_derivative(x: float):
    """ Derivative of the Logistic activation function
     :param x: Input value
     """
    f0_value = 1.0 / (1.0 + exp(-x))
    return f0_value * (1 - f0_value)


__logistic.derivative = __logistic_derivative


##################################################################

def __exponential(x: float):
    return exp(x)


__exponential.derivative = lambda x: exp(x)


##################################################################

def __reciprocal(x: float):
    return 1.0 / x


__reciprocal.derivative = lambda x: -(1.0 / x) ** 2


##################################################################

def __square(x: float):
    return x ** 2


__square.derivative = lambda x: 2 * x


##################################################################

def __gauss(x: float):
    return exp(-x ** 2)


__gauss.derivative = lambda x: -2 * x * exp(-x ** 2)


##################################################################

def __sine(x: float):
    return sin(x)


__sine.derivative = lambda x: cos(x)


##################################################################

def __cosine(x: float):
    return cos(x)


__cosine.derivative = lambda x: -sin(x)


##################################################################

def __elliot(x: float):
    return x / (1 + abs(x))


def __elliot_derivative(x: float):
    aux = fabs(x) + 1
    return (aux + x) / (aux ** 2)


__elliot.derivative = __elliot_derivative


##################################################################

def __arctan(x: float):
    return 2.0 * atan(x) / pi


__arctan.derivative = lambda x: 2.0 / ((1 + x ** 2) * pi)

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
