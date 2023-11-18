import numpy as np
from scipy.optimize import minimize
from decompose import decomposeQP
from utils import FrobeniusNorm, is_wholenumber
import random

def findSigExposures(M, P, decomposition_method=decomposeQP):
    # Process and check function parameters
    # M, P
    if M.shape[0] != P.shape[0]:
        raise ValueError("Matrices 'M' and 'P' must have the same number of rows (mutations types).")

    if P.shape[1] == 1:
        raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    # decomposition.method
    if not callable(decomposition_method):
        raise ValueError("Parameter 'decomposition_method' must be a function.")

    # Normalize M by column (just in case it is not normalized)
    M = M / M.sum(axis=0)

    # Find solutions
    # Matrix of signature exposures per sample/patient (column)
    exposures = np.apply_along_axis(decomposition_method, 0, M, P)

    # Compute estimation error for each sample/patient (Frobenius norm)
    errors = np.vectorize(lambda i: FrobeniusNorm(M[:, i], P, exposures[:, i]))(range(M.shape[1]))

    return exposures, errors



def bootstrapSigExposures(m, P, R, mutation_count=None, decomposition_method=decomposeQP):
    # Process and check function parameters
    # m, P

    P = np.array(P)
    if len(m) != P.shape[0]:
        raise ValueError("Length of vector 'm' and number of rows of matrix 'P' must be the same.")
    #if not np.all(np.array(list(m.keys())) == P.shape[0]):
    #    raise ValueError("Elements of vector 'm' and rows of matrix 'P' must have the same names (mutations types).")
    if P.shape[1] == 1:
        raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    # If 'mutation_count' is not specified, 'm' has to contain counts
    if mutation_count is None: #@todo best
        if all(is_wholenumber(val) for val in m):
            mutation_count = m.sum()
        else:
            raise ValueError("Please specify the parameter 'mutation_count' in the function call or provide mutation counts in parameter 'm'.")

    # Normalize m to be a vector of probabilities.
    m = m / np.sum(m)

    # Find optimal solutions using provided decomposition method for each bootstrap replicate
    # Matrix of signature exposures per replicate (column)
    K = len(m)  # number of mutation types

    def bootstrap_sample(m, mutation_count, K):
        mutations_sampled = random.choices(range(m.shape[0]), k=mutation_count, weights=m)
        # mutations_sampled = list(np.genfromtxt('output/mutations_sampled.csv', delimiter=',', skip_header=1))
        # np.savetxt('output/mutations_sampled.csv', mutations_sampled, delimiter=',')
        m_sampled = {k: mutations_sampled.count(k) / mutation_count for k in range(1, K + 1)}
        return list(m_sampled.values())

    exposures = np.column_stack([
        decomposition_method(bootstrap_sample(m, mutation_count, K), P) for _ in range(R)
    ])
    exposures = exposures / np.sum(exposures, axis=0)  # Normalize exposures

    # Compute estimation error for each replicate/trial (Frobenius norm)
    # G x R
    errors = np.vectorize(lambda i: FrobeniusNorm(m, P, exposures[:, i]))(range(exposures.shape[1]))


    return exposures, errors


def crossValidationSigExposures(m, P, num_folds, decomposition_method=decomposeQP):
    # Process and check function parameters
    P = np.array(P)

    if len(m) != P.shape[0]:
        raise ValueError("Length of vector 'm' and number of rows of matrix 'P' must be the same.")

    if P.shape[1] == 1:
        raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    m = m / np.sum(m)

    fold_size = len(m) // num_folds
    folds = [m[i:i + fold_size] for i in range(0, len(m), fold_size)]

    def calculate_fold_exposures(i, num_folds):
        fold = np.concatenate([folds[j] if j != i else [0] * len(folds[i] + 1) for j in range(num_folds)])
        normalized_fold = fold / fold.sum()
        return normalized_fold

    # Perform cross-validation for each replicate
    fold_exposures = np.column_stack([
        decomposition_method(calculate_fold_exposures(i, num_folds), P)
        for i in range(num_folds)
    ])
    fold_exposures = fold_exposures / np.sum(fold_exposures, axis=0)

    # Compute estimation error for each replicate/trial (Frobenius norm)
    errors = np.vectorize(lambda i: FrobeniusNorm(m, P, fold_exposures[:, i]))(range(fold_exposures.shape[1]))

    return fold_exposures, errors

