"""Test that all public API symbols can be imported."""

# All imports in this file exist solely to verify they don't error.
# ruff: noqa: F401

from src.models import (
    BaseModel,
    LeNet,
    MyNet,
    AlexNet,
    BottleneckViT,
    FeaturePyramidViT,
    FeaturePyramidViTTiny,
    FeaturePyramidViTSmall,
    FeaturePyramidViTLarge,
    SiameseFPNViT,
    SiameseFPNViTTiny,
    SiameseFPNViTSmall,
    SiameseFPNViTLarge,
    FeaturePyramidMoEViT,
    FeaturePyramidMoEViTTiny,
    FeaturePyramidMoEViTSmall,
    FeaturePyramidMoEViTLarge,
    SiameseFPNMoEViT,
    SiameseFPNMoEViTTiny,
    SiameseFPNMoEViTSmall,
    SiameseFPNMoEViTLarge,
    SiameseNetwork,
    ModelRegistry,
)
from src.datasets import (
    BaseDataset,
    ClassificationDataset,
    TripletDatasetBase,
    BalancedTripletDataset,
    MNISTDataset,
    CIFARDataset,
    Subset631Dataset,
    Subset1000Dataset,
    TripletSubset1000Dataset,
    TripletMNISTDataset,
    BalancedTripletMNISTDataset,
    DatasetRegistry,
)
from src.config import Config, load_config, create_config_parser, get_default_config
from src.training import Trainer, MetricsTracker, CheckpointManager, ExperimentManager
from src.utils import get_device, get_optimal_workers, setup_logger


def test_all_imports():
    """All imports above should succeed without errors."""
    assert True


def test_model_registry_has_all_models():
    """Every registered model should be available via ModelRegistry."""
    models = ModelRegistry.list_available()
    expected = [
        "lenet",
        "mynet",
        "alexnet",
        "bottleneck_vit",
        "fpn_vit",
        "fpn_vit_tiny",
        "fpn_vit_small",
        "fpn_vit_large",
        "siamese",
        "siamese_fpn_vit",
        "siamese_fpn_vit_tiny",
        "siamese_fpn_vit_small",
        "siamese_fpn_vit_large",
        "fpn_moe_vit",
        "fpn_moe_vit_tiny",
        "fpn_moe_vit_small",
        "fpn_moe_vit_large",
        "siamese_fpn_moe_vit",
        "siamese_fpn_moe_vit_tiny",
        "siamese_fpn_moe_vit_small",
        "siamese_fpn_moe_vit_large",
    ]
    for name in expected:
        assert name in models, f"Model '{name}' missing from registry"
    assert len(models) == len(expected), f"Expected {len(expected)} models, got {len(models)}"


def test_dataset_registry_has_all_datasets():
    """Every registered dataset should be available via DatasetRegistry."""
    datasets = DatasetRegistry.list_available()
    expected = [
        "mnist",
        "cifar10",
        "subset_631",
        "subset_1000",
        "triplet_mnist",
        "balanced_triplet_mnist",
        "triplet_subset_1000",
    ]
    for name in expected:
        assert name in datasets, f"Dataset '{name}' missing from registry"
    assert len(datasets) == len(expected), f"Expected {len(expected)} datasets, got {len(datasets)}"


def test_config_module_functions():
    """Configuration module functions should all work."""
    parser = create_config_parser()
    assert parser is not None

    default = get_default_config()
    assert default["dataset"]["name"] == "mnist"
    assert default["model"]["name"] == "mynet"
    assert default["checkpointing"]["save_frequency"] == 10
    assert default["optimization"]["scheduler"] == "none"
