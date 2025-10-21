"""Tests for utility functions in pyslingshot.util module."""

from __future__ import annotations

import numpy as np
import pytest

from pyslingshot.util import infer_cluster_label_indices, isint, scale_to_range


class TestScaleToRange:
    """Tests for scale_to_range function."""

    def test_scale_to_range_default(self):
        """Test default scaling to [0, 1]."""
        x = np.array([0.0, 5.0, 10.0])
        result = scale_to_range(x)

        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 1.0)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_scale_to_range_custom(self):
        """Test scaling to custom range."""
        x = np.array([0.0, 5.0, 10.0])
        result = scale_to_range(x, a=-1, b=1)

        assert np.isclose(result[0], -1.0)
        assert np.isclose(result[-1], 1.0)
        assert np.all(result >= -1) and np.all(result <= 1)


class TestIsint:
    """Tests for isint function."""

    def test_isint_with_int(self):
        """Test isint with actual integer."""
        assert isint(5) is True
        assert isint(0) is True
        assert isint(-10) is True

    def test_isint_with_float(self):
        """Test isint with float values."""
        assert isint(5.0) is True  # Integer value as float
        assert isint(5.5) is False  # Non-integer float

    def test_isint_with_numpy(self):
        """Test isint with numpy types."""
        assert isint(np.int64(5)) is True
        assert isint(np.float64(5.0)) is True
        assert isint(np.float64(5.5)) is False


class TestInferClusterLabelIndices:
    """Tests for infer_cluster_label_indices function."""

    def test_infer_with_int_labels(self):
        """Test inference with integer labels."""
        cluster_labels = np.array([0, 1, 2, 1, 0, 2])
        result = infer_cluster_label_indices(cluster_labels)

        np.testing.assert_array_equal(result, cluster_labels)

    def test_infer_with_str_labels(self):
        """Test inference with string labels."""
        cluster_labels = np.array(["A", "B", "C", "B", "A", "C"])
        result = infer_cluster_label_indices(cluster_labels)

        # Should convert to indices 0, 1, 2
        expected = np.array([0, 1, 2, 1, 0, 2])
        np.testing.assert_array_equal(result, expected)

    def test_infer_with_str_labels_ordering(self):
        """Test that string labels are mapped consistently."""
        cluster_labels = np.array(["cell_A", "cell_B", "cell_A", "cell_C"])
        result = infer_cluster_label_indices(cluster_labels)

        # Check consistency
        assert result[0] == result[2]  # Same label should have same index
        assert len(np.unique(result)) == 3  # Should have 3 unique indices

    def test_infer_invalid_type(self):
        """Test that invalid label types raise ValueError."""
        cluster_labels = np.array([1.5, 2.5, 3.5])  # Float labels (not integer values)

        with pytest.raises(ValueError, match="Unexpected cluster label dtype"):
            infer_cluster_label_indices(cluster_labels)
