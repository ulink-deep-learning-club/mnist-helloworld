"""Shared fixtures and configuration for framework tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Small batch size for fast testing."""
    return 4


@pytest.fixture
def dummy_input_1ch_28():
    """Dummy 1-channel 28x28 input batch."""
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def dummy_input_3ch_32():
    """Dummy 3-channel 32x32 input batch."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def dummy_input_3ch_64():
    """Dummy 3-channel 64x64 input batch."""
    return torch.randn(4, 3, 64, 64)
