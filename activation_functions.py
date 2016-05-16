from math import tanh, exp, cos, sin, fabs, atan, pi


def __default(x):
    """ Default activation function"""
    return tanh(x)


__default.derivative = lambda x: 1 - tanh(x) ** 2


def __tanh(x):
    """ Tanh activation function
     :param x: Input value
     """
    return tanh(x)


__tanh.derivative = lambda x: 1 - tanh(x) ** 2


def __identity(x):
    """ Identity activation function
     :param x: Input value
     """
    return x


__identity.derivative = lambda x: 1


def __threshold(x):
    """ Threshold activation function
    :param x: Input value
    """
    return 1 if x > 0 else 0


__threshold.derivative = lambda x: 0


def __logistic(x):
    """ Logistic activation function
    :param x: Input value
    """
    return 1.0 / (1.0 + exp(-x))


def __logistic_derivative(x):
    """ Derivative of the Logistic activation function
     :param x: Input value
     """
    f0_value = 1.0 / (1.0 + exp(-x))
    return f0_value * (1 - f0_value)


__logistic.derivative = __logistic_derivative


def __exponential(x):
    """ Exponential activation function
    :param x: Input value
    """
    return exp(x)


__exponential.derivative = lambda x: exp(x)


def __reciprocal(x):
    """ Reciprocal activation function
    :param x: Input value
    """
    return 1.0 / x


__reciprocal.derivative = lambda x: -(1.0 / x) ** 2


def __square(x):
    """ Square activation function
    :param x: Input value
    """
    return x ** 2


__square.derivative = lambda x: 2 * x


def __gauss(x):
    """ Gauss activation function
    :param x: Input value
    """
    return exp(-x ** 2)


__gauss.derivative = lambda x: -2 * x * exp(-x ** 2)


def __sine(x):
    """ Sine activation function
     :param x: Input value
     """
    return sin(x)


__sine.derivative = lambda x: cos(x)


def __cosine(x):
    """ Cosine activation function
    :param x: Input value
    """
    return cos(x)


__cosine.derivative = lambda x: -sin(x)


def __elliot(x):
    def elliot_derivative(x):
        aux = fabs(x) + 1
        return (aux + x) / (aux ** 2)

    __elliot.derivative = elliot_derivative(x)

    return x / (1 + abs(x))


def __arctan(x):
    """ Arctan activation function
    :param x: Input value
    """
    return 2.0 * atan(x) / pi


__arctan.derivative = lambda x: 2.0 / ((1 + x ** 2) * pi)

activation_functions_set = {'Tanh': __tanh,
                            'Identity': __identity,
                            'Threshold': __threshold,
                            'Logistic': __logistic,
                            'Exponential': __exponential,
                            'Reciprocal': __reciprocal,
                            'Square': __square,
                            'Sine': __sine,
                            'Cosine': __cosine,
                            'Elliot': __elliot,
                            'Arctan': __arctan,
                            'default': __default}
