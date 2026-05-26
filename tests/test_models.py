"""Test model creation, forward pass, and consistency."""

import pytest
import torch
from src.models import ModelRegistry
from src.models.base import BaseModel


# Map each model to its expected input shape and model type
MODEL_TEST_CASES = [
    # (model_name, input_channels, input_size, expected_model_type)
    ("lenet", 1, (28, 28), "classification"),
    ("mynet", 1, (28, 28), "classification"),
    ("alexnet", 1, (28, 28), "classification"),
    ("bottleneck_vit", 1, (64, 64), "classification"),
    ("fpn_vit", 1, (64, 64), "classification"),
    ("fpn_vit_tiny", 1, (64, 64), "classification"),
    ("fpn_vit_small", 1, (64, 64), "classification"),
    ("fpn_vit_large", 1, (64, 64), "classification"),
    ("siamese", 1, (28, 28), "siamese"),
]


class TestModelCreation:
    """Models should be creatable from the registry with correct properties."""

    @pytest.mark.parametrize("name,channels,size,mtype", MODEL_TEST_CASES)
    def test_create_and_forward(self, name, channels, size, mtype):
        model = ModelRegistry.create(
            name,
            num_classes=10,
            input_channels=channels,
            input_size=size,
        )
        assert isinstance(model, BaseModel), f"{name} is not a BaseModel"
        assert model.model_type == mtype, f"{name}.model_type should be '{mtype}'"
        assert model.num_classes == 10
        assert model.input_channels == channels

        # Forward pass
        x = torch.randn(2, channels, *size)
        output = model(x)
        assert isinstance(output, torch.Tensor), f"{name} output should be a tensor"
        assert output.shape[0] == 2, f"{name} output batch size should be 2"

    def test_create_all_registered_models(self):
        """All registered models should be creatable with minimal kwargs."""
        for name in ModelRegistry.list_available():
            model = ModelRegistry.create(
                name,
                num_classes=10,
                input_channels=1,
                input_size=(64, 64),
            )
            assert isinstance(model, BaseModel), f"{name} is not a BaseModel"
            info = model.get_model_info()
            assert info["total_parameters"] > 0, f"{name} has no parameters"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.create("nonexistent")


class TestSiameseModel:
    """Siamese model specific tests."""

    def test_forward_returns_embedding(self):
        from src.models import SiameseNetwork
        model = SiameseNetwork(num_classes=10, input_channels=1, input_size=(28, 28), embedding_dim=128)
        x = torch.randn(4, 1, 28, 28)
        model.train()
        emb = model(x)
        assert emb.shape == (4, 128), f"Expected (4, 128), got {emb.shape}"
        # Should be L2-normalized during training
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Training embeddings should be L2-normalized"

    def test_forward_with_classifier(self):
        from src.models import SiameseNetwork
        model = SiameseNetwork(num_classes=10, input_channels=1, input_size=(28, 28), embedding_dim=128)
        x = torch.randn(4, 1, 28, 28)
        logits = model.forward_with_classifier(x)
        assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"


class TestModelProperties:
    """All models should report consistent properties."""

    def test_classification_models_have_criterion_and_metrics(self):
        """Classification models should provide CrossEntropyLoss and MetricsTracker."""
        for name in ModelRegistry.list_available():
            model_class = ModelRegistry.get(name)
            try:
                model = ModelRegistry.create(name, num_classes=10, input_channels=1, input_size=(64, 64))
            except Exception:
                continue

            criterion = model_class.get_criterion()
            assert criterion is not None, f"{name}.get_criterion() returned None"

            metrics = model_class.get_metrics_tracker()
            assert metrics is not None, f"{name}.get_metrics_tracker() returned None"

            if hasattr(model, "model_type"):
                assert model.model_type in BaseModel.MODEL_TYPES, f"{name}.model_type '{model.model_type}' invalid"

    def test_moe_models_report_arch_type(self):
        """MoE models should report arch_type='moe' and has_aux_loss=True."""
        moe_names = [n for n in ModelRegistry.list_available() if "moe" in n]
        for name in moe_names:
            model = ModelRegistry.create(name, num_classes=10, input_channels=1, input_size=(64, 64))
            assert model.arch_type == "moe", f"{name}.arch_type should be 'moe'"
            assert model.has_aux_loss, f"{name}.has_aux_loss should be True"

    def test_non_moe_models_report_dense(self):
        """Non-MoE models should report arch_type='dense' and has_aux_loss=False."""
        non_moe = [n for n in ModelRegistry.list_available() if "moe" not in n]
        for name in non_moe:
            model = ModelRegistry.create(name, num_classes=10, input_channels=1, input_size=(64, 64))
            assert model.arch_type == "dense", f"{name}.arch_type should be 'dense' (got '{model.arch_type}')"
            assert not model.has_aux_loss, f"{name}.has_aux_loss should be False"
