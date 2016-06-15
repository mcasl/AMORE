import unittest

from pyAmore.baseline.activation_functions import *


class TestActivationFunctions(unittest.TestCase):
    def test_tanh(self):
        # function
        test_function = activation_functions_set['tanh']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), tanh(0.5))
        # derivative
        self.assertEqual(test_function.derivative(0, 0.0), 1.0)
        self.assertEqual(test_function.derivative(0.5, tanh(0.5)), 1 - tanh(0.5) ** 2)

    def test_identity(self):
        # function
        test_function = activation_functions_set['identity']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), 0.5)
        # derivative
        self.assertEqual(test_function.derivative(0, None), 1.0)
        self.assertEqual(test_function.derivative(0.5, None), 1.0)

    def test_threshold(self):
        # function
        test_function = activation_functions_set['threshold']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), 1)
        # derivative
        self.assertEqual(test_function.derivative(-0.5, None), 0.0)
        self.assertEqual(test_function.derivative(0.5, None), 0.0)

    def test_logistic(self):
        # function
        test_function = activation_functions_set['logistic']
        self.assertEqual(test_function(0), 1.0 / (1.0 + exp(0)))
        self.assertEqual(test_function(0.5), 1.0 / (1.0 + exp(-0.5)))
        # derivative
        self.assertEqual(test_function.derivative(0, 1.0 / (1.0 + exp(0))), 0.25)
        self.assertEqual(test_function.derivative(0.5, 1.0 / (1.0 + exp(-0.5))),
                         (1.0 / (1.0 + exp(0.5))) * (1 - (1.0 / (1.0 + exp(0.5)))))

    def test_exponential(self):
        # function
        test_function = activation_functions_set['exponential']
        self.assertEqual(test_function(0), 1.0)
        self.assertEqual(test_function(0.5), exp(0.5))
        # derivative
        self.assertEqual(test_function.derivative(0, 1.0), 1.0)
        self.assertEqual(test_function.derivative(0.5, exp(0.5)), exp(0.5))

    def test_reciprocal(self):
        # function
        test_function = activation_functions_set['reciprocal']
        self.assertEqual(test_function(1), 1.0)
        self.assertEqual(test_function(0.5), 2)
        # derivative
        self.assertEqual(test_function.derivative(0.1, 10), -100)
        self.assertEqual(test_function.derivative(0.5, 2), -4)

    def test_square(self):
        # function
        test_function = activation_functions_set['square']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), 0.25)
        # derivative
        self.assertEqual(test_function.derivative(0.1, None), 0.2)
        self.assertEqual(test_function.derivative(0.5, None), 1.0)

    def test_gauss(self):
        # function
        test_function = activation_functions_set['gauss']
        self.assertEqual(test_function(0), 1)
        self.assertEqual(test_function(0.5), exp(-0.5 ** 2))
        # derivative
        self.assertEqual(test_function.derivative(0.0, 1.0), 0)
        self.assertEqual(test_function.derivative(0.5, exp(-0.5 ** 2)), -2 * 0.5 * exp(-0.5 ** 2))

    def test_sine(self):
        # function
        test_function = activation_functions_set['sine']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), sin(0.5))
        # derivative
        self.assertEqual(test_function.derivative(0.0, None), 1.0)
        self.assertEqual(test_function.derivative(0.5, None), cos(0.5))

    def test_cosine(self):
        # function
        test_function = activation_functions_set['cosine']
        self.assertEqual(test_function(0), 1.0)
        self.assertEqual(test_function(0.5), cos(0.5))
        # derivative
        self.assertEqual(test_function.derivative(0.0, None), 0.0)
        self.assertEqual(test_function.derivative(0.5, None), -sin(0.5))

    def test_elliot(self):
        # function
        test_function = activation_functions_set['elliot']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(-0.5), -0.5 / (1 + abs(-0.5)))
        # derivative
        self.assertEqual(test_function.derivative(0.0, None), ((fabs(0) + 1) + 0) / ((fabs(0) + 1) ** 2))
        self.assertEqual(test_function.derivative(0.5, None), ((fabs(0.5) + 1) + 0.5) / ((fabs(0.5) + 1) ** 2))

    def test_arctan(self):
        # function
        test_function = activation_functions_set['arctan']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), 2 * atan(0.5) / pi)
        # derivative
        self.assertEqual(test_function.derivative(0.0, None), 2.0 / ((1 + 0.0 ** 2) * pi))
        self.assertEqual(test_function.derivative(0.5, None), 2.0 / ((1 + 0.5 ** 2) * pi))

    def test_default(self):
        # function
        test_function = activation_functions_set['default']
        self.assertEqual(test_function(0), 0.0)
        self.assertEqual(test_function(0.5), tanh(0.5))
        # derivative
        self.assertEqual(test_function.derivative(0, 0.0), 1.0)
        self.assertEqual(test_function.derivative(0.5, tanh(0.5)), 1 - tanh(0.5) ** 2)


if __name__ == '__main__':
    unittest.main()
