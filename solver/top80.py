import cvxpy as cp
import numpy as np


def solve(x0, w):
    n = len(x0)
    x = cp.Variable(n)
    constraints = []
    objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS")
    return x.value
