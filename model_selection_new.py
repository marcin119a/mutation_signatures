import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
from decompose import decomposeQP
from utils import is_wholenumber
def compute_p_value(exposures, threshold=0.01):
    grater_than_threshold = exposures > threshold

    return 1 - grater_than_threshold.sum(axis=1) / grater_than_threshold.shape[1]


def bootstraped_patient(m, mutation_count, R):
    K = len(m)

    if mutation_count is None:
        if all(is_wholenumber(val) for val in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError("Please specify the parameter 'mutation_count' in the function call or provide mutation counts in parameter 'm'.")
    m = m / np.sum(m)


    def bootstrap_sample(m, mutation_count, K):
        mutations_sampled = np.random.choice(K, size=mutation_count, p=m)
        return np.bincount(mutations_sampled, minlength=K) / mutation_count

    M = np.column_stack([bootstrap_sample(m, mutation_count, K) for _ in range(R)])
    return M


def backward_elimination(
    m, P, R, threshold, mutation_count, significance_level, decomposition_method=decomposeQP
):
    best_columns = np.arange(P.shape[1])
    P_temp = P
    model_error, _ = findSigExposures(
        m.reshape(-1, 1), P_temp, decomposition_method
    )
    M = bootstraped_patient(m, mutation_count, R)

    while True:
        changed = False

        exposures, errors = findSigExposures(
            M, P_temp, decomposition_method=decomposition_method
        )
        p_values = compute_p_value(exposures, threshold=threshold)

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

import cProfile
import pstats

def main_block():
    tumorBRCA = np.genfromtxt('data/M.csv', delimiter='\t', skip_header=1)
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)[:, :2]

    signaturesCOSMIC = np.genfromtxt('output/WGS_signatures__sigProfiler_SBS_signatures_2019_05_22.modified.csv', delimiter='\t', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)

    for i in range(tumorBRCA.shape[1]):
        first_col = tumorBRCA[:, i]

        best_columns, b, estimation_exposures = backward_elimination(first_col, signaturesCOSMIC, threshold=0.01, mutation_count=10000, R=1000, significance_level=0.01)
        print(best_columns)

if __name__ == '__main__':
    #cProfile.runctx('main_block()', globals(), locals(), 'profile_results.prof')
    #s = pstats.Stats('profile_results.prof')
    #s.strip_dirs().sort_stats('time').print_stats()
    print(main_block())