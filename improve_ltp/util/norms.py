import numpy as np


class L2Norm:
    def __init__(self):
        self.fun = np.linalg.norm

    def __call__(self, vec):
        return self.fun(vec)
