import numpy as np
import unittest

from decompose import decomposeQP  # Upewnij się, że importujesz funkcję z właściwego modułu
from utils import FrobeniusNorm

class TestDecomposeQP(unittest.TestCase):
    def test_decomposeQP(self):
        # Przykładowe dane wejściowe
        m = np.array([0.5, 0.3, 0.2])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])

        expected_exposures = np.array([0.2, 0.3, 0.5])

        exposures = decomposeQP(m, P)
        print(exposures)
        #np.testing.assert_array_almost_equal(exposures, expected_exposures, decimal=6)

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



if __name__ == '__main__':
    unittest.main()
