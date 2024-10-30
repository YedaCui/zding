import numpy as np


def solve(x0, w):
    return np.where(
        x0 > 1,
        x0 - 1,
        np.where(
            x0 < -1,
            x0 + 1,
            x0,
        ),
    )
