import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split

from .base import BaseDataset


class Subset631Dataset(BaseDataset):
    """Subset 631 Chinese character dataset."""

    @property
    def dataset_type(self) -> str:
        return "standard"

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = True,
    ):
        super().__init__(root, download, reapply_transforms)
        self._train_indices = None
        self._test_indices = None

    def get_train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def load_data(self):
        """Load Subset 631 dataset."""

        data_path = os.path.join(self.root, "subset_631")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Use fixed seed for reproducible split
        import torch

        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        # Store indices to ensure consistent splits across reloads
        self._train_indices = train_dataset.indices
        self._test_indices = test_dataset.indices

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

    @property
    def num_classes(self) -> int:
        return 631

    @property
    def input_channels(self) -> int:
        return 1

    def get_index_label_mapping(self) -> dict:
        """Get mapping from class index to character label.

        Returns:
            dict: Mapping from integer index to character string.
        """
        if self._train_dataset is None:
            self.load_data()

        # ImageFolder stores class_to_idx mapping
        full_dataset = self._train_dataset.dataset
        idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
        return idx_to_class

    def export_index_label_json(self, output_path: str = "index_label_mapping.json"):
        """Export index-label mapping to JSON file.

        Args:
            output_path: Path to save the JSON file.
        """
        import json

        mapping = self.get_index_label_mapping()

        def decode_label(label):
            """Decode label to proper Chinese character."""
            if isinstance(label, str):
                # Check for #UXXXX format (common in some datasets)
                if label.startswith("#U") or label.startswith("#u"):
                    try:
                        hex_code = label[2:]  # Remove #U or #u prefix
                        return chr(int(hex_code, 16))
                    except (ValueError, OverflowError):
                        return label
                # Check for Unicode escape sequences like \u4e14
                if "\\u" in label or "\\U" in label:
                    try:
                        return label.encode("utf-8").decode("unicode-escape")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        return label
                # Already a proper character
                return label
            return str(label)

        # Convert int keys to strings and decode labels
        mapping_str_keys = {str(k): decode_label(v) for k, v in mapping.items()}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping_str_keys, f, ensure_ascii=False, indent=2)

        print(f"Index-label mapping exported to {output_path}")
        return output_path

    @property
    def input_size(self) -> tuple:
        return (64, 64)

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        data_path = os.path.join(self.root, "subset_631")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        # Use stored indices to maintain same train/test split
        if self._train_indices is not None:
            self._train_dataset = Subset(full_dataset, self._train_indices)
        else:
            # Fallback: create new split (shouldn't happen in normal usage)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            import torch

            generator = torch.Generator().manual_seed(42)
            self._train_dataset, _ = random_split(
                full_dataset, [train_size, test_size], generator=generator
            )
