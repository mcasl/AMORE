from math import tanh, exp, cos, sin, fabs, atan, pi, nan


def default_f0(*args):
    """ Default activation function"""
    return nan


def default_f1(*args):
    """ Derivative of the default activation function """
    return nan


def tanh_f0(x):
    """ Tanh activation function """
    return tanh(x)


def tanh_f1(x):
    """ Derivative of the Tanh activation function """
    return 1 - tanh(x) ** 2


def identity_f0(x):
    """ Identity activation function """
    return x


def identity_f1(*args):
    """ Derivative of the Identity activation function """
    return 1


def threshold_f0(x):
    """ Threshold activation function """
    return 1 if x > 0 else 0


def threshold_f1(*args):
    """ Derivative of the Threshold activation function """
    return 0


def logistic_f0(x):
    """ Logistic activation function"""
    return 1.0 / (1.0 + exp(-x))


def logistic_f1(x):
    """ Derivative of the Logistic activation function """
    f0_value = 1.0 / (1.0 + exp(-x))
    return f0_value * (1 - f0_value)


def exponential_f0(x):
    """ Exponential activation function"""
    return exp(x)


def exponential_f1(x):
    """ Derivative of the Exponential activation function """
    return exp(x)


def reciprocal_f0(x):
    """ Reciprocal activation function"""
    return 1.0 / x


def reciprocal_f1(x):
    """ Derivative of the Reciprocal activation function """
    return -(1.0 / x) ** 2


def square_f0(x):
    """ Square activation function"""
    return x ** 2


def square_f1(x):
    """ Derivative of the Square activation function """
    return 2 * x


def gauss_f0(x):
    """ Gauss activation function"""
    return exp(-x ** 2)


def gauss_f1(x):
    """ Derivative of the Gauss activation function """
    return -2 * x * exp(-x ** 2)


def sine_f0(x):
    """ Sine activation function """
    return sin(x)


def sine_f1(x):
    """ Derivative of the Sine activation function """
    return cos(x)


def cosine_f0(x):
    """ Cosine activation function"""
    return cos(x)


def cosine_f1(x):
    """ Derivative of the Cosine activation function """
    return -sin(x)


def elliot_f0(x):
    """ Elliot activation function"""
    return x / (1 + abs(x))


def elliot_f1(x):
    """ Derivative of the Elliot activation function """

    aux = fabs(x) + 1
    return (aux + x) / (aux ** 2)


def arctan_f0(x):
    """ Arctan activation function"""
    return 2.0 * atan(x) / pi


def arctan_f1(x):
    """ Derivative of the Arctan activation function """
    return 2.0 / ((1 + x ** 2) * pi)


activation_function_set = {'Tanh': (tanh_f0, tanh_f1),
                           'Identity': (identity_f0, identity_f1),
                           'Threshold': (threshold_f0, threshold_f1),
                           'Logistic': (logistic_f0, logistic_f1),
                           'Exponential': (exponential_f0, exponential_f1),
                           'Reciprocal': (reciprocal_f0, reciprocal_f1),
                           'Square': (square_f0, square_f1),
                           'Sine': (sine_f0, sine_f1),
                           'Cosine': (cosine_f0, cosine_f1),
                           'Elliot': (elliot_f0, elliot_f1),
                           'Arctan': (arctan_f0, arctan_f1),
                           'default': (default_f0, default_f1),
                           }
