import unittest
from cost_functions import *
from math import log


class TestCostFunctions(unittest.TestCase):
    def test_lms(self):
        # function
        test_function = cost_functions_set['LMS']
        self.assertEqual(test_function(0.1, 0.1), 0.0)
        self.assertEqual(test_function(0.1, 0.6), 0.5 ** 2)
        # derivative
        self.assertEqual(test_function.derivative(0.1, 0.1), 0.0)
        self.assertEqual(test_function.derivative(0.1, 0.6), -0.5)

    def test_lmls(self):
        # function
        test_function = cost_functions_set['LMLS']
        self.assertEqual(test_function(0.1, 0.1), 0.0)
        self.assertEqual(test_function(0.1, 0.6), log(1 + 0.5 ** 2 / 2))
        # derivative
        self.assertEqual(test_function.derivative(0.1, 0.1), 0.0)
        self.assertEqual(test_function.derivative(0.1, 0.6), -0.5 / 1.125)
