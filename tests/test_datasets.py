"""Test dataset creation and transform correctness."""

import pytest
from src.datasets import DatasetRegistry
from src.datasets.base import BaseDataset
from src.datasets.mnist import MNISTDataset
from src.datasets.cifar import CIFARDataset


class TestDatasetRegistry:
    """Dataset registry should create valid dataset instances."""

    def test_create_mnist(self):
        ds = DatasetRegistry.create("mnist", image_size=28, output_channels=1)
        assert ds.num_classes == 10
        assert ds.input_channels == 1
        assert ds.input_size == (28, 28)
        assert ds.dataset_type == "standard"

    def test_create_cifar10(self):
        ds = DatasetRegistry.create("cifar10", image_size=32, output_channels=3)
        assert ds.num_classes == 10
        assert ds.input_channels == 3
        assert ds.input_size == (32, 32)
        assert ds.dataset_type == "standard"

    def test_create_subset_1000(self):
        ds = DatasetRegistry.create("subset_1000", image_size=64, output_channels=1)
        assert ds.num_classes == 1000
        assert ds.input_channels == 1

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            DatasetRegistry.create("nonexistent")

    def test_all_datasets_have_required_properties(self):
        """Every registered dataset must have a valid base class and properties."""
        for name in DatasetRegistry.list_available():
            ds = DatasetRegistry.create(name, image_size=64, output_channels=1)
            assert isinstance(ds, BaseDataset), f"{name} is not a BaseDataset"
            assert ds.num_classes > 0, f"{name}.num_classes is not > 0"
            assert ds.input_channels >= 1, f"{name}.input_channels is not >= 1"
            h, w = ds.input_size
            assert h > 0 and w > 0, f"{name}.input_size is invalid"
            assert ds.dataset_type in BaseDataset.DATASET_TYPES, f"{name}.dataset_type is invalid"


class TestCIFARDatasetTransforms:
    """CIFARDataset transform logic should be correct for both 1ch and 3ch."""

    def test_output_channels_3_no_grayscale(self):
        """When output_channels=3, CIFAR images stay RGB (no Grayscale transform)."""
        ds = CIFARDataset(output_channels=3)
        t = ds.get_train_transform(output_channels=3)
        names = [type(x).__name__ for x in t.transforms]
        assert "Grayscale" not in names, "Grayscale should NOT be applied when output_channels=3"

    def test_output_channels_1_has_grayscale(self):
        """When output_channels=1, Grayscale transform should be applied."""
        ds = CIFARDataset(output_channels=1)
        t = ds.get_train_transform(output_channels=1)
        names = [type(x).__name__ for x in t.transforms]
        assert "Grayscale" in names, "Grayscale should be applied when output_channels=1"

    def test_output_channels_3_uses_rgb_norm(self):
        """When output_channels=3, use RGB normalization stats."""
        ds = CIFARDataset(output_channels=3)
        t = ds.get_train_transform(output_channels=3)
        norm = t.transforms[-1]
        import torchvision.transforms as T
        assert isinstance(norm, T.Normalize)
        assert len(norm.mean) == 3, "RGB normalization should have 3 mean values"
        import numpy as np
        assert np.allclose(norm.mean, (0.4914, 0.4822, 0.4465)), f"Unexpected mean: {norm.mean}"

    def test_output_channels_1_uses_mnist_norm(self):
        """When output_channels=1, use MNIST (grayscale) normalization stats."""
        ds = CIFARDataset(output_channels=1)
        t = ds.get_train_transform(output_channels=1)
        norm = t.transforms[-1]
        import torchvision.transforms as T
        assert isinstance(norm, T.Normalize)
        assert len(norm.mean) == 1, "Grayscale normalization should have 1 mean value"
        assert norm.mean[0] == pytest.approx(0.1307, abs=1e-4), f"Unexpected mean: {norm.mean}"

    def test_internal_train_transform_consistent(self):
        """The _train_transform stored during __init__ should be consistent."""
        ds = CIFARDataset(output_channels=3)
        names = [type(x).__name__ for x in ds._train_transform.transforms]
        assert "Grayscale" not in names

        ds1 = CIFARDataset(output_channels=1)
        names1 = [type(x).__name__ for x in ds1._train_transform.transforms]
        assert "Grayscale" in names1


class TestMNISTDatasetDefaults:
    """MNISTDataset should have correct defaults."""

    def test_default_channels(self):
        ds = MNISTDataset()
        assert ds.output_channels == 1
        assert ds.input_channels == 1

    def test_default_image_size(self):
        ds = MNISTDataset()
        assert ds.image_size == 28
        assert ds.input_size == (28, 28)
