import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
from utils import calculate_BIC
from decompose import decomposeQP

def runBootstrapOnMatrix(
    M, P, R, mutation_count=1000, threshold=0.01, decomposition_method=decomposeQP
):
    def process_column(column):
        exposures, errors = bootstrapSigExposures(
            column, P, R, mutation_count, decomposition_method
        )
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]



def runCrossvaldiationOnMatrix(
    M, P, fold_size=4, threshold=0.01, decomposition_method=decomposeQP
):
    def process_column(column):
        exposures, errors = crossValidationSigExposures(
            column,
            P,
            fold_size,
            True,
            decomposition_method,
        )
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, M)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]



def backward_elimination(
    m, P, R, threshold, mutation_count, significance_level, decomposition_method=decomposeQP
):
    best_columns = np.arange(P.shape[1])
    P_temp = P
    model_error, _ = findSigExposures(
        m.reshape(-1, 1), P_temp, decomposition_method
    )
    while True:
        changed = False
        p_values = runBootstrapOnMatrix(
            m,
            P_temp,
            R,
            mutation_count=mutation_count,
            threshold=threshold,
            decomposition_method=decomposition_method,
        )

        max_p_value = p_values.max()
        if max_p_value > significance_level:
            max_p_var = p_values.argmax()
            best_columns = np.delete(best_columns, max_p_var)
            P_temp = P[:, best_columns]
            exposures, errors = findSigExposures(
                m.reshape(-1, 1),
                P_temp,
                decomposition_method=decomposition_method,
            )
            print(exposures.sum(axis=0))
            print(calculate_BIC(P_temp, exposures, errors))
            changed = True

        if not changed:
            break

    return (
        best_columns,
        bootstrapSigExposures(
            m,
            P_temp,
            mutation_count=1000,
            R=R,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(-1, 1), P_temp, decomposition_method=decomposition_method
        ),
    )



def forward_elimination(
    m, P, R, threshold, mutation_count, significance_level, decomposition_method=decomposeQP
):
    best_columns = [1]
    P_temp = np.zeros((P.shape[0], 0))
    while True:
        changed = False
        best_p_value = 1
        best_p_var = -1

        for i in range(P.shape[1] - 1, 0, -1):
            if i in best_columns:
                continue

            # Try adding the ith column to the model
            P_temp_with_i = np.hstack([P_temp, P[:, [i]]])
            p_values = runBootstrapOnMatrix(
                m, P_temp_with_i, R, mutation_count=1000, threshold=0.05
            )

            # Check if the new variable is significant and better than the current best
            if p_values[-1] < best_p_value and p_values[-1] < significance_level:
                best_p_value = p_values[-1]
                best_p_var = i

        # If a significant variable was found, add it to the model
        if best_p_var != -1:
            best_columns.append(best_p_var)
            P_temp = np.hstack([P_temp, P[:, [best_p_var]]])
            changed = True

        # Stop if no new significant variable was found
        if not changed:
            break

    return (
        best_columns,
        bootstrapSigExposures(
            m,
            P_temp,
            mutation_count=1000,
            R=R,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(m.shape[0], 1), P_temp, decomposition_method=decomposition_method
        ),
    )



#to test
if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('data/counts_119breast.csv', delimiter=',', skip_header=1)
    patients = np.genfromtxt('data/counts_119breast.csv', delimiter=',', max_rows=1, dtype=str)[1:]
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    signaturesCOSMIC = np.genfromtxt('tests/data/signaturesCOSMIC.csv', delimiter=',', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    first_col = tumorBRCA[:, 0]
    spec = [ x-1 for x in  [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]]

    _, _, _ = backward_elimination(first_col, signaturesCOSMIC, R=10, significance_level=0.01)
