import unittest
import numpy as np
from estimates_exposures_test import findSigExposures

class TestEstimateExposures(unittest.TestCase):
    def test_findSigExposures(self):
        M = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])

        result = findSigExposures(M, P)

        expected_exposures = np.array([[0.6097561, 0.3902439], [0.6097561, 0.3902439], [0.6097561, 0.3902439]])
        expected_errors = np.array([0.08049845, 0.08049845, 0.08049845])

        np.testing.assert_array_almost_equal(result['exposures'], expected_exposures, decimal=7)
        np.testing.assert_array_almost_equal(result['errors'], expected_errors, decimal=7)



if __name__ == '__main__':
    unittest.main()
