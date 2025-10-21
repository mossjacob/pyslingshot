from __future__ import annotations

from typing import Any

import numpy as np


def mahalanobis(
    X1: np.ndarray,
    X2: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
) -> np.ndarray:
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)


def isint(x: Any) -> bool:
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


def scale_to_range(x: np.ndarray, a: float = 0, b: float = 1) -> np.ndarray:
    return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a


def infer_cluster_label_indices(cluster_labels: np.ndarray) -> np.ndarray:
    """
    Infers the cluster label indices: a vector of
    Args:
        cluster_labels: list of string labels or list of integers denoting cluster assignment
    Returns: np.array of cluster assignments
    """
    if isint(cluster_labels[0]):
        return cluster_labels
    elif isinstance(cluster_labels[0], str):
        # Convert list of str labels into a list of int indices
        return np.array([np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels])
    else:
        raise ValueError("Unexpected cluster label dtype.")
