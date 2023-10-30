import numpy as np
import unittest

from decompose import decomposeQP  # Upewnij się, że importujesz funkcję z właściwego modułu

class TestDecomposeQP(unittest.TestCase):
    def test_decomposeQP(self):
        m = np.array([0.5, 0.3, 0.2])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])

        expected_exposures = np.array([0.2, 0.3, 0.5])

        exposures = decomposeQP(m, P)
        print(exposures)
        #np.testing.assert_array_almost_equal(exposures, expected_exposures, decimal=6)

if __name__ == '__main__':
    unittest.main()
