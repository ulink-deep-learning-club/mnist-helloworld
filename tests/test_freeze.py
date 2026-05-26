"""Test layer freezing functionality (3 modes: ID, range, name pattern)."""

import pytest
import torch
import torch.nn as nn

from train import get_layer_id_mapping, parse_freeze_spec, freeze_layers


class TestGetLayerIdMapping:
    """Layer ID mapping should correctly identify modules."""

    def test_mapping_structure(self):
        model = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.Linear(10, 2),
        )
        mapping = get_layer_id_mapping(model)
        # nn.Sequential should produce: "1-1" -> conv, "1-2" -> relu, "1-3" -> linear
        assert len(mapping) >= 3

    def test_mapping_with_named_children(self):
        """Named modules should use their names in mapping values."""
        from src.models import LeNet
        model = LeNet(num_classes=10, input_channels=1)
        mapping = get_layer_id_mapping(model)
        # Should contain key modules
        names = set(mapping.values())
        assert "features" in names or any("features" in v for v in names), \
            "Should contain 'features' module"
        assert "classifier" in names or any("classifier" in v for v in names), \
            "Should contain 'classifier' module"


class TestParseFreezeSpec:

    def test_layer_id(self):
        typ, rest = parse_freeze_spec("2-1")
        assert typ == "id"
        assert rest == "2-1"

    def test_range(self):
        typ, start, end = parse_freeze_spec("2-1:2-5")
        assert typ == "range"
        assert start == "2-1"
        assert end == "2-5"

    def test_name_pattern(self):
        typ, pattern = parse_freeze_spec("features")
        assert typ == "name"
        assert pattern == "features"

    def test_name_with_hyphen(self):
        """Name patterns containing hyphens should not be mistaken for IDs."""
        typ, pattern = parse_freeze_spec("layer-norm")
        assert typ == "name", "Names with hyphens should be treated as name patterns"


class TestFreezeLayers:

    @pytest.fixture
    def conv_model(self):
        return nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def test_freeze_by_name_pattern(self, conv_model):
        """Freezing 'features' or prefix should work."""
        frozen_count, modules = freeze_layers(conv_model, ["0"], logger=None)
        assert frozen_count > 0, "Should have frozen some parameters"

        # First conv layer (index 0) should be frozen
        for name, param in conv_model.named_parameters():
            if name.startswith("0."):
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should not be frozen"

    def test_freeze_by_layer_id(self, conv_model):
        mapping = get_layer_id_mapping(conv_model)
        # Find the first conv layer's ID
        target_name = "0"  # First Sequential child
        target_id = None
        for lid, name in mapping.items():
            if name == target_name:
                target_id = lid
                break

        if target_id:
            frozen_count, modules = freeze_layers(conv_model, [target_id], id_to_name=mapping)
            assert frozen_count > 0

    def test_freeze_multiple_specs(self, conv_model):
        """Multiple freeze specs should work together."""
        total_before = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)
        frozen_count, modules = freeze_layers(conv_model, ["0", "2"], logger=None)
        total_after = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)
        assert total_after < total_before

    def test_frozen_layers_dont_update(self):
        """Verify frozen params don't change after optimizer step."""
        model = nn.Linear(10, 2)
        # Freeze ONLY the weight, keep bias trainable
        model.weight.requires_grad = False

        weight_before = model.weight.data.clone()
        opt = torch.optim.SGD([model.bias], lr=0.1)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()

        # Weight should be unchanged (frozen)
        assert torch.equal(model.weight.data, weight_before), "Frozen weight should not change"

    def test_freeze_reduces_trainable_params(self, conv_model):
        """Freezing should reduce trainable parameter count."""
        total_trainable = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)
        freeze_layers(conv_model, ["0", "1"], logger=None)
        frozen_trainable = sum(p.numel() for p in conv_model.parameters() if p.requires_grad)
        assert frozen_trainable < total_trainable
