import numpy as np
from estimates_exposures import crossValidationSigExposures, findSigExposures
from decompose import decomposeQP




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
    m, P, fold_size, threshold, significance_level, decomposition_method=decomposeQP
):
    best_columns = np.arange(P.shape[1])
    P_temp = P
    model_error, _ = findSigExposures(
        m.reshape(-1, 1), P_temp, decomposition_method
    )
    while True:
        changed = False
        p_values = runCrossvaldiationOnMatrix(
            m,
            P_temp,
            fold_size=fold_size,
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
        runCrossvaldiationOnMatrix(
            m,
            P_temp,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(-1, 1), P_temp, decomposition_method=decomposition_method
        ),
    )






