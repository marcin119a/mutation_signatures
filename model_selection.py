import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
from decompose import decomposeQP

def runBootstrapOnMatrix(
    m, P, R, mutation_count, threshold=0.01, decomposition_method=decomposeQP
):
    def process_column(column):
        exposures, errors = bootstrapSigExposures(
            column, P, R, mutation_count, decomposition_method
        )
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, m)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]



def runCrossvaldiationOnMatrix(
    m, P, fold_size=4, threshold=0.01, decomposition_method=decomposeQP
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

    all_exposures = np.apply_along_axis(process_column, 0, m)

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
            indices_with_max = np.where(p_values == max_p_value)[0]
            max_p_var = np.random.choice(indices_with_max)
            best_columns = np.delete(best_columns, max_p_var)
            P_temp = P[:, best_columns]

            changed = True

        if not changed:
            break

    return (
        best_columns,
        bootstrapSigExposures(
            m,
            P_temp,
            mutation_count=mutation_count,
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
                m, P_temp_with_i, R, mutation_count=mutation_count, threshold=threshold
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
            mutation_count=mutation_count,
            R=R,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(m.shape[0], 1), P_temp, decomposition_method=decomposition_method
        ),
    )






