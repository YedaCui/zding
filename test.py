import importlib
import os
import sys
import time

import numpy as np
from numpy import random as R


def cost(x):
    x = np.abs(x)
    return np.where(x <= 1, 0.01, x) * x


def obj(x1, x0, w):
    return x1 @ w + cost(x1 - x0).sum() + (cost(x1.sum()) + cost(x1 @ np.sign(x0))) / 2


def gen(n):
    return R.normal(R.normal(0, 1), abs(R.normal(0, 1)), n), R.normal(0, 0.01, n)


for i, filename in enumerate(os.listdir("solver")):
    if filename == sys.argv[1] + ".py":
        cpu = i
        os.sched_setaffinity(0, [cpu])  # bind cpu to disable parallelization

solver = importlib.import_module("solver." + sys.argv[1])
R.seed(0)  # will be changed in official test
cases = [gen(max(round(1.01 ** i), i)) for i in range(1000)]
for i, (x0, w) in enumerate(cases):
    # print(x0)
    # print(w)
    t0 = time.time()
    x1 = solver.solve(x0, w)
    t1 = time.time()
    print("case", i, t1 - t0, obj(x1, x0, w))
