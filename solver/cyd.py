import cvxpy as cp
import numpy as np



def solve(x0, w):
    n = len(x0)
    x = cp.Variable(n)
    constraints = []
    objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    res = x.value
    idx0 = np.where(res - x0 <= 0)[0]
    idx1 = np.where(res - x0 > 0)[0]
    y1 = 1 if np.sum(res) >= 0 else -1 
    y2 = 1 if res @ np.sign(x0) >= 0 else -1 

    la = 100
    x = cp.Variable(n)
    constraints = []
    objective_terms = [x @ w + cp.square(x - x0).sum() + 0.5 * (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0)))]
    # Add terms based on idx0 and idx1 if they are not empty
    if idx0.size > 0:
        objective_terms.append(la * cp.norm1(x[idx0] - x0[idx0] + 1)) 
    if idx1.size > 0:
        objective_terms.append(la * cp.norm1(x[idx1] - x0[idx1] - 1)) 
    objective_terms.append(la * (cp.abs(cp.sum(x) - y1) + cp.abs(x @ np.sign(x0) - y2))) 
    
    objective = cp.Minimize(cp.sum(objective_terms))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value
