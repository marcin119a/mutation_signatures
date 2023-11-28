import numpy as np


def FrobeniusNorm(M, P, E):
    return np.sqrt(np.sum((M - np.dot(P, E))**2))


def is_wholenumber(x, tol=1e-15):
    return np.abs(x - np.round(x)) < tol


def calculate_BIC(M, exposures, errors):
    n = M.shape[1]  # Number of patients
    k = exposures.shape[0]  # Number of signatures
    RSS = np.sum(errors**2)

    log_likelihood = -n/2 * np.log(RSS)
    BIC = k * np.log(n) - 2 * log_likelihood

    return BIC
