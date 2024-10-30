import numpy as np


def solve(x0, w):
    return np.where(
        w > 0.01,
        x0 - 1,
        np.where(
            w < -0.01,
            x0 + 1,
            x0,
        ),
    )
