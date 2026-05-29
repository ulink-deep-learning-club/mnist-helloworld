"""Tests for device detection utilities."""

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.device import get_device, get_optimal_workers


# ========================================================================
# get_device
# ========================================================================

class TestGetDevice:
    """get_device returns correct device based on input."""

    def test_cpu_returns_cpu_device(self):
        device, using_cpu = get_device("cpu")
        assert device.type == "cpu"
        assert using_cpu is True

    def test_cpu_case_insensitive(self):
        device, using_cpu = get_device("CPU")
        assert device.type == "cpu"
        assert using_cpu is True

        device, using_cpu = get_device("Cpu")
        assert device.type == "cpu"
        assert using_cpu is True

    def test_invalid_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported device"):
            get_device("invalid")

    def test_auto_equal_none(self):
        """'auto' behaves the same as None — auto-detect."""
        # We can't easily test which device it picks without mocking,
        # but we can verify they return the same type pair.
        d1, c1 = get_device()
        d2, c2 = get_device("auto")
        assert d1.type == d2.type
        assert c1 == c2

    # --- Mocked scenarios ---

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_picks_cuda_when_available(self, mock_mps, mock_cuda):
        device, using_cpu = get_device()
        assert device.type == "cuda"
        assert using_cpu is False

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_picks_mps_when_cuda_unavailable(self, mock_mps, mock_cuda):
        device, using_cpu = get_device()
        # Note: on non-MPS systems this test verifies the logic branch;
        # torch.device("mps") works as a device constructor even without MPS hardware.
        assert device.type == "mps"
        assert using_cpu is False

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_picks_cpu_when_nothing_available(self, mock_mps, mock_cuda):
        device, using_cpu = get_device()
        assert device.type == "cpu"
        assert using_cpu is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_raises(self, mock_cuda):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            get_device("cuda")

    @patch("torch.backends.mps.is_available", return_value=False)
    def test_mps_unavailable_raises(self, mock_mps):
        with pytest.raises(RuntimeError, match="MPS is not available"):
            get_device("mps")


# ========================================================================
# get_optimal_workers
# ========================================================================

class TestGetOptimalWorkers:
    """get_optimal_workers returns sensible worker counts."""

    def test_cpu_returns_one_one(self):
        train_w, val_w = get_optimal_workers(True)
        assert train_w == 1
        assert val_w == 1

    @patch("os.cpu_count", return_value=4)
    def test_non_cpu_with_four_cores(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == min(8, max(2, 4 // 2))  # 2
        assert val_w == min(4, max(1, 2 // 2))   # 1

    @patch("os.cpu_count", return_value=8)
    def test_non_cpu_with_eight_cores(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == min(8, max(2, 8 // 2))  # 4
        assert val_w == min(4, max(1, 4 // 2))   # 2

    @patch("os.cpu_count", return_value=16)
    def test_non_cpu_with_sixteen_cores(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == min(8, max(2, 16 // 2))  # 8
        assert val_w == min(4, max(1, 8 // 2))    # 4

    @patch("os.cpu_count", return_value=32)
    def test_train_workers_caps_at_eight(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == 8  # capped
        assert val_w == 4    # capped

    @patch("os.cpu_count", return_value=2)
    def test_non_cpu_with_two_cores(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == max(2, 2 // 2)  # 2 (max floor)
        assert val_w == max(1, 2 // 2)   # 1

    @patch("os.cpu_count", return_value=1)
    def test_non_cpu_with_one_core(self, mock_cpu_count):
        train_w, val_w = get_optimal_workers(False)
        assert train_w == max(2, 1 // 2)  # 2
        assert val_w == max(1, 2 // 2)   # 1

    @patch("os.cpu_count", return_value=0)
    def test_non_cpu_with_zero_cores_fallback(self, mock_cpu_count):
        """When os.cpu_count() returns 0 or None, fallback to 1."""
        # os.cpu_count() returning 0 is not realistic, but the code handles it
        train_w, val_w = get_optimal_workers(False)
        assert train_w >= 2
        assert val_w >= 1
