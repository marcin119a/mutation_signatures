import unittest
import numpy as np
from estimates_exposures import *

class TestEstimateExposures(unittest.TestCase):
    def test_findSigExposures(self):
        M = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])

        result = findSigExposures(M, P)

        expected_exposures = np.array([[0.6097561, 0.3902439], [0.6097561, 0.3902439], [0.6097561, 0.3902439]])
        expected_errors = np.array([0.08049845, 0.08049845, 0.08049845])

        np.testing.assert_array_almost_equal(result['exposures'], expected_exposures, decimal=7)
        np.testing.assert_array_almost_equal(result['errors'], expected_errors, decimal=7)


class TestBootstrapSample(unittest.TestCase):
    def test_bootstrap_sample(self):
        m = np.array([0.5, 0.3, 0.2])
        mutation_count = 100
        K = len(m)

        result = bootstrap_sample(m, mutation_count, K)

        # Sprawdzenie, czy wynik to lista z K elementami, a suma tych elementów wynosi mutation_count
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), K)
        self.assertEqual(sum(result), mutation_count)


class TestBootstrapSigExposures(unittest.TestCase):

    def test_bootstrapSigExposures(self):
        m = np.array([0.5, 0.3, 0.2])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])
        R = 1000
        mutation_count = 10

        result = bootstrapSigExposures(m, P, R, mutation_count)
        print("Exposures:")
        print(result['exposures'])
        print("Errors:")
        print(result['errors'])

if __name__ == '__main__':
    unittest.main()
