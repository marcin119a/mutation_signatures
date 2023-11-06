from decompose import decomposeQP
from estimates_exposures import findSigExposures
import numpy as np

if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.genfromtxt('data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    first_col = tumorBRCA[:, 0]
    res = decomposeQP(first_col, signaturesCOSMIC)

    exposures, errors = findSigExposures(tumorBRCA, signaturesCOSMIC, decomposition_method=decomposeQP)
    print (exposures, errors)
    print(exposures.shape, errors.shape)
    # Zapisz exposures do pliku CSV
    np.savetxt('exposures.csv', exposures, delimiter=',')

    # Zapisz errors do pliku CSV
    np.savetxt('errors.csv', errors, delimiter=',')