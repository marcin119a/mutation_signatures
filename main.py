from decompose import decomposeQP
from estimates_exposures import findSigExposures
import numpy as np

if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', skip_header=1)
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    signaturesCOSMIC = np.genfromtxt('data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    first_col = tumorBRCA[:, 0]
    res = decomposeQP(first_col, signaturesCOSMIC)

    exposures, errors = findSigExposures(tumorBRCA, signaturesCOSMIC, decomposition_method=decomposeQP)
    print(exposures, errors)
    print(exposures.shape, errors.shape)
    np.savetxt('output/exposures.csv', exposures, delimiter=',')

    np.savetxt('output/errors.csv', errors, delimiter=',')