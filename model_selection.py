import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures

def runBootstrapOnMatrix(M, P, R, mutation_count= 1000, threshold=0.01):
    def process_column(column):
        exposures, errors = bootstrapSigExposures(column, P, R, mutation_count)
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]


def runCrossvaldiationOnMatrix(M, P, fold_size=4, threshold=0.01):
    def process_column(column):
        exposures, errors = crossValidationSigExposures(column, P, fold_size, shuffle=True)
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]


def backward_elimination(m, P, R, significance_level=0.05):
    best_columns = np.arange(P.shape[1])
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

    return bootstrapSigExposures(m, P_temp, R=R), findSigExposures(m.reshape(m.shape[0], 1), P_temp)

if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/counts_119breast.csv', delimiter=',', skip_header=1)
    patients = np.genfromtxt('data/counts_119breast.csv', delimiter=',', max_rows=1, dtype=str)[1:]
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    signaturesCOSMIC = np.genfromtxt('data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    first_col = tumorBRCA[:, 0]
    spec = [ x-1 for x in  [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]]

    result_exposures = backward_elimination(first_col, signaturesCOSMIC, R=10, significance_level=0.01)
    print(result_exposures[0][0].shape)