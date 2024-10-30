import cvxpy as cp
import numpy as np

def cost(x):
    x = np.abs(x)
    return np.where(x <= 1, 0.01, x) * x


def obj(x1, x0, w):
    return x1 @ w + cost(x1 - x0).sum() + (cost(x1.sum()) + cost(x1 @ np.sign(x0))) / 2

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

    la = 10
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


# def solve(x0, w):
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.maximum(0.01 * (cp.abs(x-x0)-1) + 1, cp.square(x-x0)).sum() \
#                                  + (cp.maximum(0.01 * (cp.abs(cp.sum(x))-1) + 1, cp.square(cp.sum(x))) \
#                                      + cp.maximum(0.01 * (cp.abs(x @ np.sign(x0))-1) + 1, cp.square(x @ np.sign(x0)))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     return x.value

def grad_cost(x):
    return np.where(np.abs(x) <= 1, 0.01, 2*np.abs(x)) * np.sign(x)

# def solve(x0, w):
#     max_iter, lr, tol = 1000, 0.1, 1e-6
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     res = x.value
#     resmax, objmax = res.copy(), obj(res, x0, w)
#     for _ in range(max_iter):
#         print(f"begin the {_}th iteration")
#         grad = w + grad_cost(res-x0) + grad_cost(res.sum())/2 + grad_cost(res @ np.sign(x0)) * np.sign(x0)/2
#         res = res - lr * grad
#         objnew = obj(res, x0, w)
#         print(grad)
#         print(res - x0)
#         print(np.sum(res))
#         print(res @ np.sign(x0))
#         if objnew < objmax:
#             resmax, objmax = res.copy(), objnew
#         if np.linalg.norm(grad) < tol:
#             break
#         input()
#     return resmax

# def solve(x0, w):
#     max_iter, lr, tol = 1000, 0.1, 1e-6
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     res = x.value
#     resmax, objmax = res.copy(), obj(res, x0, w)
#     gamma, momentum, sigma = 0.9, 0, 0.1
#     for _ in range(max_iter):
#         grad = w + grad_cost(res-x0) + grad_cost(res.sum())/2 + grad_cost(res @ np.sign(x0)) * np.sign(x0)/2
#         momentum = gamma * momentum + lr * grad
#         noise = np.random.normal(0,sigma)
#         res = res - momentum + noise
#         objnew = obj(res, x0, w)
#         if objnew < objmax:
#             resmax, objmax = res.copy(), objnew
#         sigma *= 0.99
#         if np.linalg.norm(grad) < tol:
#             break
#     return resmax


# def solve(x0, w):
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = [cp.abs(x-x0) <= 1, cp.abs(cp.sum(x)) <= 1, cp.abs(x @ np.sign(x0)) <= 1]
#     objective = cp.Minimize(x @ w + 0.01*cp.abs(x-x0).sum() + (0.01*cp.sum(x) + 0.01*x @ np.sign(x0)) / 2)
#     # objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     if not (problem.status == "infeasible"):
#         return x.value
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     return x.value

# def solve(x0, w):
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = [cp.abs(x-x0) <= 1]
#     objective = cp.Minimize(x @ w + 0.01*cp.abs(x-x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     # objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     x1, obj1 = x.value, obj(x.value, x0, w)
#     x = cp.Variable(n)
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     x2, obj2 = x.value, obj(x.value, x0, w)
#     return x1 if obj1 < obj2 else x2


# def solve(x0, w):
#     n = len(x0)
#     x = cp.Variable(n)
#     constraints = []
#     objective = cp.Minimize(x @ w + cp.square(x - x0).sum() + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     idx0 = np.where(np.abs(x.value - x0) <= 1)[0]
#     idx1 = np.where(np.abs(x.value - x0) > 1)[0]
#     x = cp.Variable(n)
#     constraints = []
#     if idx0.size > 0:
#         constraints.append(cp.abs(x[idx0] - x0[idx0]) <= 1)
#     # objective = cp.Minimize(x @ w + 0.01 * cp.norm1(x[idx0] - x0[idx0]) + cp.sum_squares(x[idx1] - x0[idx1]) + (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0))) / 2)
#     # Define the objective function components
#     objective_terms = [x @ w,
#                        0.5 * (cp.square(cp.sum(x)) + cp.square(x @ np.sign(x0)))]
    
#     # Add terms based on idx0 and idx1 if they are not empty
#     if idx0.size > 0:
#         objective_terms.append(0.01 * cp.norm1(x[idx0] - x0[idx0]))  # Sum of absolute differences for idx0
#     if idx1.size > 0:
#         objective_terms.append(cp.sum_squares(x[idx1] - x0[idx1]))   # Sum of squares for idx1
    
#     # Combine all objective terms
#     objective = cp.Minimize(cp.sum(objective_terms))
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver="SCS")
#     return x.value

# def solve(x0,w):
#     n = len(x0)
#     res = np.linalg.inv(2*np.eye(n) + np.ones((n,1)) @ np.ones((1,n)) + np.where(x0>=0,1,-1).reshape(-1,1) @ np.where(x0>=0,1,-1).reshape(1,-1)) @ (2*x0 - w).reshape(-1,1)
#     return res.ravel()