import numpy as np
from scipy.optimize import minimize
from utils import FrobeniusNorm
import quadprog

def decomposeQPOLD(m, P):
    N = P.shape[1]

    def objective(E):
        return FrobeniusNorm(m, P, E)

    constraints = ({'type': 'eq', 'fun': lambda E: np.array([np.sum(E) - 1])})
    bounds = [(0, None)] * N

    out = minimize(objective, np.zeros(N), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 10, 'disp': True})

    exposures = out.x
    exposures[exposures < 0] = 0
    exposures = exposures / np.sum(exposures)

    return exposures



def decomposeQP(m, P):
    # N: how many signatures are selected
    N = P.shape[1]
    # G: matrix appearing in the quadratic programming objective function
    G = np.dot(P.T, P).astype(float)
    # C: matrix constraints under which we want to minimize the quadratic programming objective function.
    C = np.column_stack([np.ones(N), np.eye(N)]).astype(float)
    # b: vector containing the values of b_0.
    b = np.array([1] + [0]*N).astype(float)
    # d: vector appearing in the quadratic programming objective function
    d = np.dot(m.T, P).astype(float)

    # Solve quadratic programming problem
    out = quadprog.solve_qp(G, d, C, b, meq=1)

    # Some exposure values are negative, but very close to 0
    # Change these negative values to zero and renormalize
    exposures = out[0]
    exposures[exposures < 0] = 0
    exposures /= sum(exposures)

    # return the exposures
    return exposures

