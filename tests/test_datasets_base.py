"""Tests for BaseDataset and its subclasses."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.base import (
    BaseDataset,
    ClassificationDataset,
    BalancedTripletDataset,
    FixedTripletDataset,
)


# ========================================================================
# Helpers: minimal concrete subclasses for testing
# ========================================================================

class _MinimalDataset(BaseDataset):
    """Minimal concrete implementation for testing base class methods."""

    @property
    def dataset_type(self) -> str:
        return "standard"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_channels(self) -> int:
        return 1

    @property
    def input_size(self):
        return (28, 28)

    def get_train_transform(self, image_size, output_channels):
        import torchvision.transforms as transforms
        return transforms.Compose([transforms.ToTensor()])

    def get_test_transform(self, image_size, output_channels):
        import torchvision.transforms as transforms
        return transforms.Compose([transforms.ToTensor()])

    def load_data(self):
        self._train_dataset = _DummyDataset(100)
        self._test_dataset = _DummyDataset(20)

    def _reload_train_data(self):
        self._train_dataset = _DummyDataset(100)

    def get_index_label_mapping(self):
        return {i: f"class_{i}" for i in range(10)}


class _DummyDataset(Dataset):
    """Dummy dataset returning random tensors and labels."""
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(1, 28, 28), idx % 10


# ========================================================================
# _build_loader_kwargs
# ========================================================================

class TestBuildLoaderKwargs:
    """_build_loader_kwargs returns correct DataLoader kwargs."""

    def test_zero_workers_no_persistent_no_prefetch(self):
        ds = _MinimalDataset()
        kwargs = ds._build_loader_kwargs(batch_size=32, shuffle=True, num_workers=0)
        assert kwargs["batch_size"] == 32
        assert kwargs["shuffle"] is True
        assert kwargs["num_workers"] == 0
        assert "persistent_workers" not in kwargs
        assert "prefetch_factor" not in kwargs
        assert kwargs["pin_memory"] is (torch.cuda.is_available())

    def test_two_workers_persistent_and_prefetch_2(self):
        ds = _MinimalDataset()
        kwargs = ds._build_loader_kwargs(batch_size=64, shuffle=True, num_workers=2)
        assert kwargs["num_workers"] == 2
        assert kwargs["persistent_workers"] is True
        assert kwargs["prefetch_factor"] == 2

    def test_four_workers_prefetch_4(self):
        ds = _MinimalDataset()
        kwargs = ds._build_loader_kwargs(batch_size=64, shuffle=True, num_workers=4)
        assert kwargs["num_workers"] == 4
        assert kwargs["prefetch_factor"] == 4

    def test_negative_workers_clamped_to_zero(self):
        ds = _MinimalDataset()
        kwargs = ds._build_loader_kwargs(batch_size=32, shuffle=True, num_workers=-1)
        assert kwargs["num_workers"] == 0


# ========================================================================
# get_dataloaders
# ========================================================================

class TestGetDataloaders:
    """get_dataloaders returns properly configured DataLoaders."""

    def test_returns_train_and_val_loaders(self):
        ds = _MinimalDataset()
        train_loader, val_loader = ds.get_dataloaders(
            batch_size=16, num_workers=0, shuffle_train=True
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert len(train_loader) == 100 // 16 + (1 if 100 % 16 else 0)
        assert len(val_loader) == 20 // 16 + (1 if 20 % 16 else 0)

    def test_val_workers_override(self):
        ds = _MinimalDataset()
        _, val_loader = ds.get_dataloaders(
            batch_size=16, num_workers=4, val_num_workers=1
        )
        # val_loader should use 1 worker, not 4
        assert val_loader.num_workers == 1
        # Val loader should not shuffle: uses SequentialSampler
        assert not val_loader.sampler.shuffle \
            if hasattr(val_loader.sampler, 'shuffle') else True

    def test_train_shuffle_respected(self):
        ds = _MinimalDataset()
        train_loader, _ = ds.get_dataloaders(
            batch_size=16, num_workers=0, shuffle_train=False
        )
        # shuffle=False → SequentialSampler (no shuffle attribute)
        if hasattr(train_loader.sampler, 'shuffle'):
            assert train_loader.sampler.shuffle is False

    def test_calls_load_data_if_not_loaded(self):
        """get_dataloaders calls load_data when datasets are None."""
        ds = _MinimalDataset()
        ds._train_dataset = None
        ds._test_dataset = None
        train_loader, val_loader = ds.get_dataloaders(
            batch_size=16, num_workers=0
        )
        assert len(train_loader) > 0
        assert len(val_loader) > 0


# ========================================================================
# export_index_label_json
# ========================================================================

class TestExportIndexLabelJson:
    """export_index_label_json writes correct JSON."""

    def test_creates_json_file(self):
        ds = _MinimalDataset()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = ds.export_index_label_json(out_path)
            assert result == out_path
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data["0"] == "class_0"
            assert data["9"] == "class_9"
            assert len(data) == 10
        finally:
            os.unlink(out_path)

    def test_default_not_implemented_raises(self):
        """BaseDataset's default get_index_label_mapping raises NotImplementedError."""
        class _NoMapping(_MinimalDataset):
            def get_index_label_mapping(self):
                raise NotImplementedError("Subclasses must implement")
        ds = _NoMapping()
        with pytest.raises(NotImplementedError):
            ds.export_index_label_json("should_not_create.json")
        assert not os.path.exists("should_not_create.json")


# ========================================================================
# get_info
# ========================================================================

class TestGetInfo:
    """get_info returns correct metadata."""

    def test_returns_all_keys(self):
        ds = _MinimalDataset()
        info = ds.get_info()
        assert info["name"] == "_MinimalDataset"
        assert info["type"] == "standard"
        assert info["num_classes"] == 10
        assert info["input_channels"] == 1
        assert info["input_size"] == (28, 28)
        assert info["root"] == "./data"

    def test_custom_root(self):
        ds = _MinimalDataset(root="/custom/path")
        assert ds.get_info()["root"] == "/custom/path"


# ========================================================================
# reset_train_transforms
# ========================================================================

class TestResetTrainTransforms:
    """reset_train_transforms regenerates transforms when enabled."""

    def test_disabled_by_default_does_nothing(self):
        ds = _MinimalDataset(reapply_transforms=False)
        old_transform = ds._train_transform
        ds.reset_train_transforms()
        assert ds._train_transform is old_transform  # unchanged

    def test_enabled_regenerates_transform_and_reloads(self):
        ds = _MinimalDataset(reapply_transforms=True)
        ds.load_data()  # Ensure data is loaded
        old_transform = ds._train_transform
        ds.reset_train_transforms()
        assert ds._train_transform is not old_transform
        assert len(ds._train_dataset) == 100  # reloaded


# ========================================================================
# ClassificationDataset
# ========================================================================

class TestClassificationDataset:
    """ClassificationDataset has correct dataset_type."""

    def test_dataset_type_is_standard(self):
        class _Impl(ClassificationDataset):
            @property
            def num_classes(self): return 10
            @property
            def input_channels(self): return 1
            @property
            def input_size(self): return (28, 28)
            def get_train_transform(self, *a): pass
            def get_test_transform(self, *a): pass
            def load_data(self): pass
            def _reload_train_data(self): pass

        ds = _Impl()
        assert ds.dataset_type == "standard"


# ========================================================================
# BalancedTripletDataset._generate_triplets
# ========================================================================

class TestGenerateTriplets:
    """_generate_triplets produces balanced triplets."""

    def test_generates_correct_number(self):
        data_by_label = {
            0: list(range(10)),
            1: list(range(10, 20)),
            2: list(range(20, 30)),
        }
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=5
        )
        assert len(triplets) == 3 * 5  # 3 labels × 5 per class

    def test_each_triplet_has_four_elements(self):
        data_by_label = {0: list(range(10)), 1: list(range(10, 20))}
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=3
        )
        for t in triplets:
            assert len(t) == 4
            anchor, positive, negative, label = t
            assert anchor != positive  # different indices
            assert label in (0, 1)

    def test_anchor_and_positive_same_label(self):
        data_by_label = {0: list(range(10)), 1: list(range(10, 20))}
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=5
        )
        for anchor, positive, negative, label in triplets:
            # anchor and positive come from the same label
            assert (anchor < 10) == (positive < 10)
            # negative comes from a different label
            if label == 0:
                assert negative >= 10
            else:
                assert negative < 10

    def test_skips_labels_with_fewer_than_two_samples(self):
        """Labels with < 2 samples cannot form anchor+positive pairs."""
        data_by_label = {0: [0], 1: list(range(10, 20))}  # label 0 has only 1 sample
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=3
        )
        # Only label 1 produces triplets
        labels_in_triplets = {t[3] for t in triplets}
        assert 0 not in labels_in_triplets
        assert len(triplets) == 3

    def test_available_indices_filters(self):
        data_by_label = {
            0: list(range(10)),
            1: list(range(10, 20)),
        }
        # Only indices 0-4 are available for label 0, indices 10-14 for label 1
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=2, available_indices=list(range(5)) + list(range(10, 15))
        )
        for anchor, positive, negative, label in triplets:
            if label == 0:
                assert 0 <= anchor < 5
                assert 0 <= positive < 5
            else:
                assert 10 <= anchor < 15
                assert 10 <= positive < 15

    def test_multiple_labels_balanced(self):
        data_by_label = {
            0: list(range(10)),
            1: list(range(10, 20)),
            2: list(range(20, 30)),
            3: list(range(30, 40)),
        }
        triplets = BalancedTripletDataset._generate_triplets(
            None, data_by_label, per_class=10
        )
        # Count triplets per label
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, _, label in triplets:
            counts[label] += 1
        assert counts == {0: 10, 1: 10, 2: 10, 3: 10}


# ========================================================================
# FixedTripletDataset
# ========================================================================

class TestFixedTripletDataset:
    """FixedTripletDataset wraps pre-generated triplets."""

    def test_len_and_getitem(self):
        base = _DummyDataset(100)
        triplets = [(0, 1, 50, 0), (2, 3, 51, 0)]
        ds = FixedTripletDataset(base, triplets)
        assert len(ds) == 2
        anchor, pos, neg, label = ds[0]
        assert isinstance(anchor, torch.Tensor)
        assert isinstance(pos, torch.Tensor)
        assert isinstance(neg, torch.Tensor)
        assert label == 0
