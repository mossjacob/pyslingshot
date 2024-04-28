import numpy as np


def mahalanobis(X1, X2, S1, S2):
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)

def isint(x):
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        return False
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def isstr(x):
    return isinstance(x, str)


def scale_to_range(x, a=0, b=1):
    return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a


def infer_cluster_label_indices(cluster_labels):
    """
    Infers the cluster label indices: a vector of
    :param cluster_labels: list of string labels or list of integers denoting cluster assignment
    :return: np.array of cluster assignments
    """
    if isint(cluster_labels[0]):
        return cluster_labels
    elif isstr(cluster_labels[0]):
        # Convert list of str labels into a list of int indices
        return np.array([np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels])
    else:
        raise ValueError("Unexpected cluster label dtype.")
