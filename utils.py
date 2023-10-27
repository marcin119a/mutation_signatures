import numpy as np


def FrobeniusNorm(M, P, E):
    return np.sqrt(np.sum((M - np.dot(P, E))**2))


def is_wholenumber(x, tol=1e-15):
    return abs(x - round(x)) < tol
