import numpy as np
from scipy.optimize import minimize
from utils import FrobeniusNorm
import quadprog

def decomposeQPScipy(m, P):
    N = P.shape[1]

    def objective(E):
        return FrobeniusNorm(m, P, E)

    constraints = ({'type': 'eq', 'fun': lambda E: np.array([np.sum(E) - 1])})
    bounds = [(0, None)] * N

    out = minimize(
        objective, x0=np.zeros(N), method='SLSQP', bounds=bounds, constraints=constraints)


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
def decomposeQ(m, P):
    pass

"""
#from sklearn.linear_model import Ridge
def decomposeQ(m, P):
    def objective_function(beta, X, y):
        return np.sum((y - X.dot(beta)) ** 2) + alpha * np.sum(beta ** 2)

    ridge_model = Ridge(alpha=0)  # alpha is the regularization parameter
    ridge_model.fit(P, m)
    ridge_coefficients = ridge_model.coef_
    # Initial values of the coefficients
    beta_initial = ridge_coefficients

    # Optimization with constraints
    alpha = 0  # Adjustment parameter, as in the ridge regression
    result = minimize(objective_function, beta_initial, args=(P, m))

    return result.x
"""
