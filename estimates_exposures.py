import numpy as np
from scipy.optimize import minimize
from decompose import decomposeQP
from utils import FrobeniusNorm


def findSigExposures(M, P, decomposition_method=decomposeQP):
    # Process and check function parameters
    # M, P
    M = np.array(M)
    P = np.array(P)
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
    idx = np.arange(M.shape[1])
    #errors = FrobeniusNorm(M[:, idx], P, exposures[:, idx])
    errors = np.vectorize(lambda i: FrobeniusNorm(M[:, i], P, exposures[:, i]))(range(M.shape[1]))
    return exposures, errors
