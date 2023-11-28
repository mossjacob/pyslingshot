import numpy as np


def mahalanobis(X1, X2, S1, S2):
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)


def scale_to_range(x, a=0, b=1):
    return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a
