"""Pytest fixtures and shared utilities for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def data_dir() -> Path:
    """Return path to the project root directory containing test data files."""
    return Path(__file__).parent.parent


@pytest.fixture
def fakedata_1branch(data_dir: Path) -> dict[str, np.ndarray]:
    """Load the 1-branch fake data."""
    data_path = data_dir / "fakedata-1branch.npy"
    data_dict = np.load(data_path, allow_pickle=True).item()
    return data_dict


@pytest.fixture
def fakedata_2branch(data_dir: Path) -> dict[str, np.ndarray]:
    """Load the 2-branch fake data."""
    data_path = data_dir / "fakedata-2branch.npy"
    data_dict = np.load(data_path, allow_pickle=True).item()
    return data_dict


@pytest.fixture
def anndata_1branch(fakedata_1branch: dict[str, np.ndarray]) -> AnnData:
    """Create an AnnData object from 1-branch data."""
    num_cells = fakedata_1branch["data"].shape[0]
    num_genes = 500  # Arbitrary number of genes
    ad = AnnData(np.zeros((num_cells, num_genes)))
    ad.obsm["X_umap"] = fakedata_1branch["data"]
    ad.obs["celltype"] = fakedata_1branch["cluster_labels"]
    return ad


@pytest.fixture
def anndata_2branch(fakedata_2branch: dict[str, np.ndarray]) -> AnnData:
    """Create an AnnData object from 2-branch data."""
    num_cells = fakedata_2branch["data"].shape[0]
    num_genes = 500  # Arbitrary number of genes
    ad = AnnData(np.zeros((num_cells, num_genes)))
    ad.obsm["X_umap"] = fakedata_2branch["data"]
    ad.obs["celltype"] = fakedata_2branch["cluster_labels"]
    return ad


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Create simple synthetic data for basic tests."""
    np.random.seed(42)
    num_cells = 100
    num_clusters = 3

    data_list: list[np.ndarray] = []
    cluster_labels_list: list[int] = []

    for k in range(num_clusters):
        # Create clusters at different positions
        center = np.array([k * 3.0, k * 2.0])
        cluster_data = np.random.randn(num_cells // num_clusters, 2) * 0.5 + center
        data_list.append(cluster_data)
        cluster_labels_list.extend([k] * (num_cells // num_clusters))

    data = np.vstack(data_list)
    cluster_labels = np.array(cluster_labels_list)

    return data, cluster_labels


def validate_pseudotime(pseudotime: np.ndarray, num_cells: int) -> None:
    """Validate that pseudotime output has expected properties.

    Args:
        pseudotime: Pseudotime array to validate
        num_cells: Expected number of cells

    Raises:
        AssertionError: If validation fails
    """
    assert pseudotime.shape == (num_cells,), f"Expected shape ({num_cells},), got {pseudotime.shape}"
    assert np.all(np.isfinite(pseudotime)), "Pseudotime contains non-finite values"
    assert np.all(pseudotime >= 0), "Pseudotime contains negative values"


def validate_lineages(lineages: list, num_clusters: int) -> None:
    """Validate that lineages have expected properties.

    Args:
        lineages: List of Lineage objects
        num_clusters: Total number of clusters in the data

    Raises:
        AssertionError: If validation fails
    """
    assert len(lineages) > 0, "No lineages detected"
    for lineage in lineages:
        assert len(lineage) > 0, "Empty lineage detected"
        for cluster_id in lineage:
            assert 0 <= cluster_id < num_clusters, f"Invalid cluster ID {cluster_id}"


def validate_mst(tree: dict[int, list[int]], num_clusters: int) -> None:
    """Validate that MST has expected properties.

    Args:
        tree: MST represented as dictionary of children
        num_clusters: Total number of clusters

    Raises:
        AssertionError: If validation fails
    """
    assert len(tree) == num_clusters, f"Tree should have {num_clusters} nodes, got {len(tree)}"
    for node, children in tree.items():
        assert 0 <= node < num_clusters, f"Invalid node ID {node}"
        for child in children:
            assert 0 <= child < num_clusters, f"Invalid child ID {child}"
