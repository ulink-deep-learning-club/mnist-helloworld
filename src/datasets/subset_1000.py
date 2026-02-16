import os
import torch
import torchvision
from torch.utils.data import random_split, Subset
import torchvision.transforms as transforms
from .base import BaseDataset


class Subset1000Dataset(BaseDataset):
    """Subset 1000 Chinese character dataset."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
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
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_data(self):
        """Load Subset 1000 dataset."""

        data_path = os.path.join(self.root, "subset_1000")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

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
        return 1000

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def input_size(self) -> tuple:
        return (64, 64)

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        data_path = os.path.join(self.root, "subset_1000")

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
