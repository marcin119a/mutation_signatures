import numpy as np
import unittest

from decompose import decomposeQP, decomposeQPScipy
from utils import load_and_process_data

class TestDecomposeQP(unittest.TestCase):
    def test_decomposeQP(self):
        # Example input data
        m = np.array([0.5, 0.3, 0.2])
        P = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.1, 0.6]])

        exposuresFast = decomposeQP(m, P)

        exposuresSlow = decomposeQPScipy(m, P)

        np.testing.assert_array_almost_equal(exposuresFast, exposuresSlow, decimal=3)

    def test_real_data_decomposeQP(self):
        first_patient, signaturesCOSMIC = (
            load_and_process_data(patient_index=0,
                                  mutational_profiles='data/tumorBRCA.csv',
                                  predf_mutational_signatures='data/signaturesCOSMIC.csv'))

        exposuresFast = decomposeQP(first_patient, signaturesCOSMIC)
        exposuresSlow = decomposeQPScipy(first_patient, signaturesCOSMIC)

        np.testing.assert_array_almost_equal(exposuresFast, exposuresSlow, decimal=4)


if __name__ == '__main__':
    unittest.main()
