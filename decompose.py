import numpy as np
from scipy.optimize import minimize
from utils import FrobeniusNorm


def decomposeQP(m, P):
    N = P.shape[1]

    def objective(E):
        return FrobeniusNorm(m, P, E)

    constraints = ({'type': 'eq', 'fun': lambda E: np.array([np.sum(E) - 1])})
    bounds = [(0, None)] * N

    out = minimize(objective, np.zeros(N), method='SLSQP', bounds=bounds, constraints=constraints)

    exposures = out.x
    exposures[exposures < 0] = 0
    exposures = exposures / np.sum(exposures)

    return exposures