import unittest
from math import log

from amore.cost_functions import *


class TestCostFunctions(unittest.TestCase):
    def test_adapt_lms(self):
        # function
        test_function = cost_functions_set['adaptLMS']
        self.assertEqual(test_function(0.1, 0.1), 0.0)
        self.assertEqual(test_function(0.1, 0.6), 0.5 ** 2)
        # derivative
        self.assertEqual(test_function.derivative(0.1, 0.1), 0.0)
        self.assertEqual(test_function.derivative(0.1, 0.6), -0.5)

    def test_adapt_lmls(self):
        # function
        test_function = cost_functions_set['adaptLMLS']
        self.assertEqual(test_function(0.1, 0.1), 0.0)
        self.assertEqual(test_function(0.1, 0.6), log(1 + 0.5 ** 2 / 2))
        # derivative
        self.assertEqual(test_function.derivative(0.1, 0.1), 0.0)
        self.assertEqual(test_function.derivative(0.1, 0.6), -0.5 / 1.125)

    def test_batch_lms(self):
        # function
        test_function = cost_functions_set['batchLMS']
        output = np.random.rand(1000)
        target = np.random.rand(1000)
        self.assertEqual(test_function(output, output), 0.0)
        self.assertEqual(test_function(output, target), np.mean((output - target) ** 2))
        # derivative
        self.assertEqual(test_function.derivative(output, output), 0.0)
        self.assertEqual(test_function.derivative(output, target), np.mean(output - target))
