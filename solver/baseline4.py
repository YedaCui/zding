import numpy as np


def solve(x0, w):
    return x0 - np.clip(np.mean(x0), -1, 1)
