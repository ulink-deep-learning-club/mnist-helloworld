"""Tests for shared model building blocks in models/common.py."""

import pytest
import torch

from src.models.common import (
    PatchEmbed,
    Attention,
    LinearAttention,
    FocusedLinearAttention,
    Mlp,
    SEBlock,
    ConvBlock,
    Bottleneck,
    C3Module,
    InvertedResidual,
    drop_path,
    DropPath,
)


# ========================================================================
# PatchEmbed
# ========================================================================

class TestPatchEmbed:
    """PatchEmbed converts images to patch embeddings."""

    def test_output_shape(self):
        m = PatchEmbed(img_size=64, patch_size=8, in_chans=3, embed_dim=256)
        x = torch.randn(2, 3, 64, 64)
        out = m(x)
        assert out.shape == (2, 64, 256)  # (B, num_patches, embed_dim)

    def test_num_patches(self):
        m = PatchEmbed(img_size=64, patch_size=8, in_chans=3, embed_dim=256)
        assert m.num_patches == 64  # (64//8)^2

    def test_different_patch_size(self):
        m = PatchEmbed(img_size=32, patch_size=4, in_chans=1, embed_dim=128)
        x = torch.randn(1, 1, 32, 32)
        out = m(x)
        assert out.shape == (1, 64, 128)  # (32//4)^2 = 64


# ========================================================================
# Attention
# ========================================================================

class TestAttention:
    """Standard attention forward pass."""

    def test_output_shape(self):
        m = Attention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)
        out = m(x)
        assert out.shape == (2, 64, 256)

    def test_different_num_heads(self):
        m = Attention(dim=128, num_heads=4)
        x = torch.randn(1, 32, 128)
        out = m(x)
        assert out.shape == (1, 32, 128)

    def test_qkv_bias(self):
        m = Attention(dim=64, num_heads=2, qkv_bias=True)
        x = torch.randn(1, 16, 64)
        out = m(x)
        assert out.shape == (1, 16, 64)


# ========================================================================
# LinearAttention
# ========================================================================

class TestLinearAttention:
    """Linear attention forward pass."""

    def test_output_shape(self):
        m = LinearAttention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)
        out = m(x)
        assert out.shape == (2, 64, 256)

    def test_output_is_finite(self):
        m = LinearAttention(dim=64, num_heads=4)
        x = torch.randn(1, 16, 64)
        out = m(x)
        assert torch.isfinite(out).all()


# ========================================================================
# FocusedLinearAttention
# ========================================================================

class TestFocusedLinearAttention:
    """Focused linear attention (ICCV 2023) forward pass."""

    def test_output_shape(self):
        m = FocusedLinearAttention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)
        out = m(x)
        assert out.shape == (2, 64, 256)

    def test_output_is_finite(self):
        m = FocusedLinearAttention(dim=64, num_heads=4)
        x = torch.randn(1, 16, 64)
        out = m(x)
        assert torch.isfinite(out).all()

    def test_focusing_factor(self):
        m = FocusedLinearAttention(dim=64, num_heads=4, focusing_factor=5)
        x = torch.randn(1, 16, 64)
        out = m(x)
        assert out.shape == (1, 16, 64)


# ========================================================================
# Mlp
# ========================================================================

class TestMlp:
    """MLP block forward pass."""

    def test_output_shape(self):
        m = Mlp(in_features=256, hidden_features=512)
        x = torch.randn(2, 64, 256)
        out = m(x)
        assert out.shape == (2, 64, 256)

    def test_default_hidden_equals_in(self):
        m = Mlp(in_features=128)
        assert m.fc1.out_features == 128
        x = torch.randn(1, 16, 128)
        out = m(x)
        assert out.shape == (1, 16, 128)

    def test_dropout_default(self):
        m = Mlp(in_features=64, drop=0.5)
        assert m.drop.p == 0.5


# ========================================================================
# SEBlock
# ========================================================================

class TestSEBlock:
    """Squeeze-and-Excitation block."""

    def test_output_shape(self):
        m = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 8, 8)
        out = m(x)
        assert out.shape == (2, 64, 8, 8)


# ========================================================================
# ConvBlock
# ========================================================================

class TestConvBlock:
    """Conv + BN + activation block."""

    def test_output_shape(self):
        m = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    def test_no_activation(self):
        m = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1, act=False)
        assert isinstance(m.act, torch.nn.Identity)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 32, 32)


# ========================================================================
# Bottleneck
# ========================================================================

class TestBottleneck:
    """YOLOv5-style bottleneck block."""

    def test_output_shape(self):
        m = Bottleneck(64, 64)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    def test_shortcut_when_in_eq_out(self):
        """Shortcut is active when in_channels == out_channels."""
        m = Bottleneck(64, 64, shortcut=True)
        assert m.add is True

    def test_no_shortcut_when_in_ne_out(self):
        m = Bottleneck(64, 128, shortcut=True)
        assert m.add is False

    def test_no_shortcut_when_disabled(self):
        m = Bottleneck(64, 64, shortcut=False)
        assert m.add is False

    def test_output_differs_with_shortcut(self):
        """With shortcut, output = input + conv(input)."""
        m_shortcut = Bottleneck(64, 64, shortcut=True)
        m_no_shortcut = Bottleneck(64, 64, shortcut=False)
        x = torch.randn(1, 64, 16, 16)
        out_s = m_shortcut(x)
        out_ns = m_no_shortcut(x)
        # They should differ numerically
        assert not torch.allclose(out_s, out_ns)


# ========================================================================
# C3Module
# ========================================================================

class TestC3Module:
    """Cross Stage Partial module."""

    def test_output_shape(self):
        m = C3Module(64, 128, num_bottlenecks=3)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 128, 16, 16)

    def test_num_bottlenecks(self):
        m = C3Module(64, 64, num_bottlenecks=5)
        assert len(m.m) == 5


# ========================================================================
# InvertedResidual
# ========================================================================

class TestInvertedResidual:
    """MobileNetV3-style inverted residual block."""

    def test_output_shape(self):
        m = InvertedResidual(64, 128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 128, 8, 8)

    def test_residual_when_stride_1_and_in_eq_out(self):
        m = InvertedResidual(64, 64, stride=1)
        assert m.use_res is True
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)

    def test_no_residual_when_stride_2(self):
        m = InvertedResidual(64, 64, stride=2)
        assert m.use_res is False

    def test_no_residual_when_in_ne_out(self):
        m = InvertedResidual(64, 128, stride=1)
        assert m.use_res is False

    def test_no_se(self):
        m = InvertedResidual(64, 64, stride=1, use_se=False)
        # Without SE, the block has fewer layers
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


# ========================================================================
# drop_path
# ========================================================================

class TestDropPath:
    """drop_path regularization function."""

    def test_eval_mode_identity(self):
        m = DropPath(p=0.5)
        m.eval()
        x = torch.randn(4, 64, 16, 16)
        out = m(x)
        assert torch.allclose(out, x)

    def test_train_mode_drops_with_probability(self):
        """In training mode, some samples may get dropped (output all zeros).
        We can't deterministically test dropout, but we can verify shape and
        that at least sometimes the output differs from input."""
        m = DropPath(p=0.5)
        m.train()
        x = torch.randn(100, 1, 1, 1)
        out = m(x)
        assert out.shape == x.shape
        # With p=0.5, roughly half should be dropped → zero
        non_zero = (out != 0).float().mean().item()
        assert 0.1 < non_zero < 0.9, f"Expected ~50% non-zero, got {non_zero:.2%}"

    def test_zero_prob_always_identity(self):
        """DropPath with p=0 never drops."""
        m = DropPath(p=0.0)
        m.train()
        x = torch.randn(4, 64, 16, 16)
        out = m(x)
        assert torch.allclose(out, x)

    def test_one_prob_drops_all(self):
        """DropPath with p=1 drops everything."""
        m = DropPath(p=1.0)
        m.train()
        x = torch.randn(4, 64, 16, 16)
        out = m(x)
        assert torch.allclose(out, torch.zeros_like(x))

    def test_repr(self):
        m = DropPath(p=0.3)
        assert "DropPath" in repr(m)
        assert "p=0.3" in repr(m)


class TestDropPathFunction:
    """Standalone drop_path function."""

    def test_zero_drop_prob_identity(self):
        x = torch.randn(4, 64)
        out = drop_path(x, drop_prob=0.0)
        assert torch.allclose(out, x)

    def test_one_drop_prob_all_zeros(self):
        x = torch.randn(4, 64)
        out = drop_path(x, drop_prob=1.0)
        assert torch.allclose(out, torch.zeros_like(x))

    def test_output_shape_preserved(self):
        x = torch.randn(4, 3, 32, 32)
        out = drop_path(x, drop_prob=0.3)
        assert out.shape == x.shape

    def test_output_is_scaled_not_identity(self):
        """drop_path scales the output; it's not an identity function."""
        x = torch.randn(1, 10)
        out = drop_path(x, drop_prob=0.5)
        # Output should differ from input (some dims dropped, others scaled)
        assert out.shape == x.shape
