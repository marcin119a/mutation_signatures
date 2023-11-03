import numpy as np
import unittest
from utils import FrobeniusNorm

class TestFrobeniusNorm(unittest.TestCase):
    def test_frobenius_norm(self):
        M = np.array([[1, 2], [4, 5]])
        P = np.array([[1, 1], [2, 8]])
        E = np.array([[2, 0], [1, 1]])

        result = FrobeniusNorm(M, P, E)

        # Define the expected Frobenius norm value
        expected_result = 8.83176

        # Check if the result matches the expected result
        self.assertAlmostEqual(result, expected_result, places=5)
