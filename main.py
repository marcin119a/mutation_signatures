from decompose import decomposeQP
from estimates_exposures import *
import numpy as np

if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', skip_header=1)
    patients = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', max_rows=1, dtype=str)[1:]
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    signaturesCOSMIC = np.genfromtxt('data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    first_col = tumorBRCA[:, 0]
    #res = decomposeQP(first_col, signaturesCOSMIC)

    #exposures, errors = findSigExposures(tumorBRCA, signaturesCOSMIC, decomposition_method=decomposeQP)
    #print(exposures, errors)
    #print(exposures.shape, errors.shape)
    #np.savetxt('output/exposures.csv', exposures, delimiter=',', header=','.join(patients))

    #np.savetxt('output/errors.csv', errors, delimiter=',')

    exposures, errors = bootstrapSigExposures(first_col, signaturesCOSMIC, 16, 2000)

    np.savetxt('output/bootstrap_exposures.csv', exposures, delimiter=',')

    np.savetxt('output/bootstrap_errors.csv', errors, delimiter=',')
    fold_size = 4
    exposures, errors = crossValidationSigExposures(first_col, signaturesCOSMIC, fold_size, decomposition_method=decomposeQP)
    np.savetxt('output/cross_valid_exposures.csv', exposures, delimiter=',')

    np.savetxt('output/cross_valid_errors.csv', errors, delimiter=',')

