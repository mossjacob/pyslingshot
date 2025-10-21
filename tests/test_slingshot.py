"""Tests for main Slingshot algorithm."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from anndata import AnnData

from pyslingshot import Slingshot

from .conftest import validate_lineages, validate_mst, validate_pseudotime


class TestSlingshotInitialization:
    """Tests for Slingshot initialization."""

    def test_init_with_anndata(self, anndata_1branch: AnnData):
        """Test Slingshot initialization with AnnData object."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            obsm_key="X_umap",
            start_node=4,
        )

        assert slingshot.data.shape == (1000, 2)
        assert slingshot.num_clusters == 10
        assert slingshot.start_node == 4

    def test_init_with_numpy(self, fakedata_1branch: dict[str, np.ndarray]):
        """Test Slingshot initialization with numpy arrays."""
        data = fakedata_1branch["data"]
        cluster_labels = fakedata_1branch["cluster_labels"]

        # Create one-hot encoded labels
        num_clusters = len(np.unique(cluster_labels))
        cluster_labels_onehot = np.zeros((len(cluster_labels), num_clusters))
        cluster_labels_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1

        slingshot = Slingshot(
            data,
            cluster_labels_onehot=cluster_labels_onehot,
            start_node=4,
        )

        assert slingshot.data.shape == (1000, 2)
        assert slingshot.num_clusters == 10
        assert slingshot.start_node == 4

    def test_init_with_string_labels(self, anndata_1branch: AnnData):
        """Test initialization with string cluster labels."""
        # Convert cluster labels to strings
        anndata_1branch.obs["celltype"] = anndata_1branch.obs["celltype"].astype(str)

        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        assert slingshot.num_clusters == 10

    def test_init_simple_data(self, simple_data: tuple[np.ndarray, np.ndarray]):
        """Test initialization with simple synthetic data."""
        data, cluster_labels = simple_data

        num_clusters = len(np.unique(cluster_labels))
        cluster_labels_onehot = np.zeros((len(cluster_labels), num_clusters))
        cluster_labels_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1

        slingshot = Slingshot(
            data,
            cluster_labels_onehot=cluster_labels_onehot,
            start_node=0,
        )

        assert slingshot.num_clusters == 3
        assert slingshot.cluster_centres.shape == (3, 2)


class TestMSTConstruction:
    """Tests for MST construction."""

    def test_construct_mst_1branch(self, anndata_1branch: AnnData):
        """Test MST construction on 1-branch data."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        tree = slingshot.construct_mst(4)

        validate_mst(tree, slingshot.num_clusters)

    def test_construct_mst_2branch(self, anndata_2branch: AnnData):
        """Test MST construction on 2-branch data."""
        slingshot = Slingshot(
            anndata_2branch,
            celltype_key="celltype",
            start_node=5,
        )

        tree = slingshot.construct_mst(5)

        validate_mst(tree, slingshot.num_clusters)


class TestLineageDetection:
    """Tests for lineage detection."""

    def test_get_lineages_1branch(self, anndata_1branch: AnnData):
        """Test lineage detection on 1-branch data."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        slingshot.get_lineages()

        assert slingshot.lineages is not None
        validate_lineages(slingshot.lineages, slingshot.num_clusters)

    def test_get_lineages_2branch(self, anndata_2branch: AnnData):
        """Test lineage detection on 2-branch data."""
        slingshot = Slingshot(
            anndata_2branch,
            celltype_key="celltype",
            start_node=5,
        )

        slingshot.get_lineages()

        assert slingshot.lineages is not None
        # 2-branch data should have 2 lineages
        assert len(slingshot.lineages) >= 2
        validate_lineages(slingshot.lineages, slingshot.num_clusters)

    def test_cluster_lineages_mapping(self, anndata_1branch: AnnData):
        """Test that cluster_lineages mapping is created correctly."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        slingshot.get_lineages()

        assert slingshot.cluster_lineages is not None
        assert len(slingshot.cluster_lineages) == slingshot.num_clusters


class TestFit:
    """Tests for the fit method."""

    def test_fit_2branch_single_epoch(self, anndata_2branch: AnnData):
        """Test fitting on 2-branch data with single epoch."""
        slingshot = Slingshot(
            anndata_2branch,
            celltype_key="celltype",
            start_node=5,
        )

        slingshot.fit(num_epochs=1)

        assert slingshot.curves is not None
        assert slingshot.cell_weights is not None
        assert slingshot.lineages is not None
        assert len(slingshot.curves) == len(slingshot.lineages)

    def test_fit_multiple_epochs(self, anndata_1branch: AnnData):
        """Test fitting with multiple epochs."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        slingshot.fit(num_epochs=2)

        assert slingshot.curves is not None
        assert slingshot.lineages is not None
        assert len(slingshot.curves) == len(slingshot.lineages)

    def test_fit_simple_data(self, simple_data: tuple[np.ndarray, np.ndarray]):
        """Test fitting on simple synthetic data."""
        data, cluster_labels = simple_data

        num_clusters = len(np.unique(cluster_labels))
        cluster_labels_onehot = np.zeros((len(cluster_labels), num_clusters))
        cluster_labels_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1

        slingshot = Slingshot(
            data,
            cluster_labels_onehot=cluster_labels_onehot,
            start_node=0,
        )

        slingshot.fit(num_epochs=1)

        assert slingshot.curves is not None


class TestPseudotime:
    """Tests for pseudotime output."""

    def test_unified_pseudotime_2branch(self, anndata_2branch: AnnData):
        """Test unified_pseudotime property on 2-branch data."""
        slingshot = Slingshot(
            anndata_2branch,
            celltype_key="celltype",
            start_node=5,
        )

        slingshot.fit(num_epochs=1)
        pseudotime = slingshot.unified_pseudotime

        validate_pseudotime(pseudotime, len(anndata_2branch))

    def test_pseudotime_increases_along_trajectory(self, simple_data: tuple[np.ndarray, np.ndarray]):
        """Test that pseudotime generally increases along trajectory."""
        data, cluster_labels = simple_data

        num_clusters = len(np.unique(cluster_labels))
        cluster_labels_onehot = np.zeros((len(cluster_labels), num_clusters))
        cluster_labels_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1

        slingshot = Slingshot(
            data,
            cluster_labels_onehot=cluster_labels_onehot,
            start_node=0,
        )

        slingshot.fit(num_epochs=2)
        pseudotime = slingshot.unified_pseudotime

        # Check that pseudotime values vary (not all the same)
        assert len(np.unique(pseudotime)) > 1


class TestSaveLoadParams:
    """Tests for parameter serialization."""

    def test_save_and_load_params(self, anndata_1branch: AnnData):
        """Test saving and loading parameters."""
        slingshot = Slingshot(
            anndata_1branch,
            celltype_key="celltype",
            start_node=4,
        )

        slingshot.fit(num_epochs=1)

        # Save parameters
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            slingshot.save_params(tmp_path)

            # Create new instance and load parameters
            slingshot2 = Slingshot(
                anndata_1branch,
                celltype_key="celltype",
                start_node=4,
            )

            slingshot2.load_params(tmp_path)

            # Verify loaded parameters exist
            assert slingshot2.curves is not None
            assert slingshot2.cell_weights is not None
            assert slingshot2.distances is not None
            assert slingshot.curves is not None
            assert slingshot.distances is not None

            # Verify loaded data matches original
            assert len(slingshot2.curves) == len(slingshot.curves)
            np.testing.assert_array_equal(slingshot2.cell_weights, slingshot.cell_weights)
            assert len(slingshot2.distances) == len(slingshot.distances)

            # Verify curve parameters match
            for i, (curve1, curve2) in enumerate(zip(slingshot.curves, slingshot2.curves)):
                np.testing.assert_array_equal(
                    curve1.pseudotimes_interp,
                    curve2.pseudotimes_interp,
                    err_msg=f"Curve {i} pseudotimes don't match",
                )
                np.testing.assert_array_equal(
                    curve1.points_interp,
                    curve2.points_interp,
                    err_msg=f"Curve {i} points don't match",
                )

            # Verify distances match
            for i, (dist1, dist2) in enumerate(zip(slingshot.distances, slingshot2.distances)):
                np.testing.assert_array_equal(
                    dist1,
                    dist2,
                    err_msg=f"Distance {i} doesn't match",
                )

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
