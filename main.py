from decompose import decomposeQP
from estimates_exposures import findSigExposures
import pandas as pd
import numpy as np

if __name__ == '__main__':
    tumorBRCA = pd.read_csv('data/tumorBRCA.csv', index_col=0).values
    signaturesCOSMIC = pd.read_csv('data/signaturesCOSMIC.csv', index_col=0).values
    first_col = tumorBRCA[:, 0]
    res = decomposeQP(first_col, signaturesCOSMIC)

    exposures, errors = findSigExposures(tumorBRCA, signaturesCOSMIC, decomposition_method=decomposeQP)
    print (exposures, errors)
    print(exposures.shape, errors.shape)
    # Zapisz exposures do pliku CSV
    np.savetxt('exposures.csv', exposures, delimiter=',')

    # Zapisz errors do pliku CSV
    np.savetxt('errors.csv', errors, delimiter=',')