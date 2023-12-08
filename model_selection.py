import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures
def runBootstrapOnMatrix(M, P, R, mutation_count= 1000, threshold=0.01):
    def process_column(column):
        exposures, errors = bootstrapSigExposures(column, P, R, mutation_count)
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return all_exposures.sum(axis=1) / all_exposures.shape[1]

def runCrossvaldiationOnMatrix(M, P, fold_size=4, threshold=0.01):
    def process_column(column):
        exposures, errors = crossValidationSigExposures(column, P, fold_size, shuffle=True)
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return all_exposures.sum(axis=1) / all_exposures.shape[1]

def backward_elimination(m, P, significance_level=0.05):
    best_columns = np.arange(P.shape[1])
    R = 10
    P_temp = P
    while True:
        changed = False
        p_values = runBootstrapOnMatrix(m, P_temp, R, mutation_count=1000, threshold=0.01)

        max_p_value = p_values.max()
        if max_p_value > significance_level:
            max_p_var = p_values.argmax()
            best_columns = np.delete(best_columns, max_p_var)
            P_temp = P[:, best_columns]
            changed = True

        if not changed:
            break



    return best_columns

if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', skip_header=1)
    patients = np.genfromtxt('data/tumorBRCA.csv', delimiter=',', max_rows=1, dtype=str)[1:]
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    signaturesCOSMIC = np.genfromtxt('data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    first_col = tumorBRCA[:, 0]

    result_exposures = backward_elimination(first_col, signaturesCOSMIC, significance_level=0.01)
    print(result_exposures.shape)