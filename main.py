from decompose import decomposeQP
from estimates_exposures import *
from model_selection import *
import numpy as np
from utils import *

if __name__ == '__main__':
    tumorBRCA, signaturesCOSMIC = load_and_process_data(None,
                                                        'tests/data/tumorBRCA.csv',
                                                        'tests/data/signaturesCOSMIC.csv')


    #res = decomposeQP(first_col, signaturesCOSMIC)

    #exposures, errors = findSigExposures(first_col.reshape(first_col.shape[0], 1), signaturesCOSMIC, decomposition_method=decomposeQP)
    #print(exposures, errors)
    #print(exposures.shape, errors.shape)
    #np.savetxt('output/exposures.csv', exposures, delimiter=',', header=','.join(patients))

    #np.savetxt('output/errors.csv', errors, delimiter=',')
    #print(calculate_BIC(signaturesCOSMIC, exposures, errors))

    result_exposures = runCrossvaldiationOnMatrix(tumorBRCA, signaturesCOSMIC, threshold=0.01)

    np.savetxt('output/perturbed_patient2.csv', result_exposures.sum(axis=1), delimiter=',')
    #exposures, errors = bootstrapSigExposures(first_col, signaturesCOSMIC, 16, 2000)
    #np.savetxt('output/bootstrap_exposures.csv', exposures, delimiter=',')

    #np.savetxt('output/bootstrap_errors.csv', errors, delimiter=',')
    #fold_size = 5
    #exposures, errors = crossValidationSigExposures(first_col, signaturesCOSMIC, fold_size, decomposition_method=decomposeQP)
    #np.savetxt('output/cross_valid_exposures.csv', exposures, delimiter=',')

    #np.savetxt('output/cross_valid_errors.csv', errors, delimiter=',')

