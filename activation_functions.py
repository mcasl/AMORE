from math import tanh, exp, cos, sin, fabs, atan, pi, nan


def default_f0(*args):
    return nan


def default_f1(*args):
    return nan


def tanh_f0(x):
    return tanh(x)


def tanh_f1(x):
    return 1 - tanh(x) ** 2


def identity_f0(x):
    return x


def identity_f1(*args):
    return 1


def threshold_f0(x):
    return 1 if x > 0 else 0


def threshold_f1(*args):
    return 0


def logistic_f0(x):
    return 1.0 / (1.0 + exp(-x))


def logistic_f1(x):
    f0_value = 1.0 / (1.0 + exp(-x))
    return f0_value * (1 - f0_value)


def exponential_f0(x):
    return exp(x)


def exponential_f1(x):
    return exp(x)


def reciprocal_f0(x):
    return 1.0 / x


def reciprocal_f1(x):
    return -(1.0 / x) ** 2


def square_f0(x):
    return x ** 2


def square_f1(x):
    return 2 * x


def gauss_f0(x):
    return exp(-x ** 2)


def gauss_f1(x):
    return -2 * x * exp(-x ** 2)


def sine_f0(x):
    return sin(x)


def sine_f1(x):
    return cos(x)


def cosine_f0(x):
    return cos(x)


def cosine_f1(x):
    return -sin(x)


def elliot_f0(x):
    return x / (1 + abs(x))


def elliot_f1(x):
    aux = fabs(x) + 1
    return (aux + x) / (aux ** 2)


def arctan_f0(x):
    return 2.0 * atan(x) / pi


def arctan_f1(x):
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
