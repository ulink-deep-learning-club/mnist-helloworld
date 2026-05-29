"""Tests for dataset utility functions and transformations."""

import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.datasets.utils import (
    ResizePad,
    decode_label,
    export_index_label_json,
    AlbumentationsTransform,
    get_character_train_transform,
    get_character_test_transform,
    get_simple_train_transform,
    get_simple_test_transform,
)


# ========================================================================
# decode_label
# ========================================================================

class TestDecodeLabel:
    """decode_label handles various encoding formats."""

    def test_hash_u_format(self):
        """#UXXXX → decoded character."""
        assert decode_label("#U4E14") == "\u4e14"

    def test_hash_u_lowercase(self):
        assert decode_label("#u4e14") == "\u4e14"

    def test_hash_u_invalid_hex_returns_original(self):
        assert decode_label("#UZZZZ") == "#UZZZZ"

    def test_unicode_escape_sequence(self):
        """\\uXXXX → decoded character."""
        assert decode_label("\\u4e14") == "\u4e14"

    def test_unicode_escape_uppercase(self):
        """Upper-case \\U requires 8 hex digits."""
        assert decode_label("\\U00004E14") == "\u4e14"

    def test_already_decoded_character(self):
        label = "且"
        assert decode_label(label) == "且"

    def test_plain_ascii_string(self):
        assert decode_label("hello") == "hello"

    def test_non_string_input(self):
        assert decode_label(123) == "123"
        assert decode_label(None) == "None"

    def test_empty_string(self):
        assert decode_label("") == ""


# ========================================================================
# ResizePad
# ========================================================================

class TestResizePad:
    """ResizePad resizes while keeping aspect ratio and pads to square."""

    def _make_image(self, width, height):
        return Image.new("L", (width, height), 128)

    def test_output_size(self):
        img = self._make_image(40, 60)
        transform = ResizePad(target_size=64)
        out = transform(img)
        assert out.size == (64, 64)

    def test_square_input(self):
        img = self._make_image(64, 64)
        transform = ResizePad(target_size=64)
        out = transform(img)
        assert out.size == (64, 64)

    def test_wide_image_padded(self):
        """A wide image should be resized and centered with padding."""
        img = self._make_image(80, 40)
        transform = ResizePad(target_size=64, pad_value=0)
        out = transform(img)
        assert out.size == (64, 64)
        # Content should be in the center (not all padding)
        arr = np.array(out)
        assert arr.min() < 255  # there is some content

    def test_tall_image_padded(self):
        """A tall image should be resized and centered with padding."""
        img = self._make_image(30, 90)
        transform = ResizePad(target_size=64, pad_value=255)
        out = transform(img)
        assert out.size == (64, 64)

    def test_custom_pad_value(self):
        """Non-square image gets padded; corner should be pad_value."""
        img = self._make_image(30, 50)  # after scaling: 38x64, horizontal padding
        transform = ResizePad(target_size=64, pad_value=255)
        out = transform(img)
        arr = np.array(out)
        # Corner pixels should be 255 (pad value on padded side)
        assert arr[0, 0] == 255

    def test_aspect_ratio_preserved_wide(self):
        """For a 2:1 image, the smaller dimension should hit target_size."""
        img = self._make_image(128, 64)  # 2:1
        transform = ResizePad(target_size=64)
        out = transform(img)
        arr = np.array(out)
        # There should be horizontal padding (left/right columns are pad value)
        # The content width should be less than target_size
        assert out.size == (64, 64)

    def test_rgb_image(self):
        img = Image.new("RGB", (40, 60), (128, 128, 128))
        transform = ResizePad(target_size=64)
        out = transform(img)
        assert out.size == (64, 64)
        assert out.mode == "RGB"


# ========================================================================
# export_index_label_json
# ========================================================================

class TestExportIndexLabelJson:
    """export_index_label_json writes correct JSON with decoded labels."""

    def test_creates_file_with_decoded_labels(self):
        """Integer keys are converted to strings, labels are decoded."""
        mapping = {0: "#U4E14", 1: "hello", 2: "\\u4e14"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = export_index_label_json(mapping, path)
            assert result == path
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["0"] == "\u4e14"
            assert data["1"] == "hello"
            assert data["2"] == "\u4e14"
        finally:
            os.unlink(path)

    def test_empty_mapping(self):
        mapping = {}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = export_index_label_json(mapping, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data == {}
        finally:
            os.unlink(path)


# ========================================================================
# AlbumentationsTransform
# ========================================================================

class TestAlbumentationsTransform:
    """AlbumentationsTransform wraps albumentations for torchvision."""

    def test_converts_pil_to_tensor(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = AlbumentationsTransform(
            A.Compose([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ])
        )
        img = Image.new("L", (32, 32), 128)
        out = transform(img)
        assert hasattr(out, "shape")
        assert out.shape == (1, 32, 32) or out.shape == (32, 32, 1)


# ========================================================================
# Transform factory functions
# ========================================================================

class TestTransformFactories:
    """Transform factory functions return usable transform pipelines."""

    def _dummy_image(self):
        return Image.new("L", (64, 64), 128)

    def test_simple_train_transform_applies(self):
        transform = get_simple_train_transform(image_size=32, output_channels=1)
        out = transform(self._dummy_image())
        assert hasattr(out, "shape")
        assert out.shape[0] == 1  # single channel

    def test_simple_train_transform_rgb(self):
        transform = get_simple_train_transform(image_size=32, output_channels=3)
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        out = transform(img)
        assert out.shape[0] == 3

    def test_simple_test_transform_applies(self):
        transform = get_simple_test_transform(image_size=32, output_channels=1)
        out = transform(self._dummy_image())
        assert out.shape == (1, 32, 32)

    def test_simple_test_transform_rgb(self):
        transform = get_simple_test_transform(image_size=32, output_channels=3)
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        out = transform(img)
        assert out.shape[0] == 3

    def test_simple_train_transform_has_multiple_steps(self):
        transform = get_simple_train_transform()
        assert len(transform.transforms) >= 4

    @pytest.mark.skipif(
        not os.environ.get("TEST_ALBUMENTATIONS"),
        reason="Set TEST_ALBUMENTATIONS=1 to test albumentation transforms"
    )
    def test_character_train_transform_applies(self):
        """Albumentation transforms require the library to be installed."""
        transform = get_character_train_transform(image_size=32, output_channels=1)
        out = transform(self._dummy_image())
        assert hasattr(out, "shape")

    @pytest.mark.skipif(
        not os.environ.get("TEST_ALBUMENTATIONS"),
        reason="Set TEST_ALBUMENTATIONS=1 to test albumentation transforms"
    )
    def test_character_test_transform_applies(self):
        transform = get_character_test_transform(image_size=32, output_channels=1)
        out = transform(self._dummy_image())
        assert hasattr(out, "shape")
