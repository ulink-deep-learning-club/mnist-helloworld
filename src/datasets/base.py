from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    DATASET_TYPES = ["standard", "pairwise", "triplet"]

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
    ):
        self.root = root
        self.download = download
        self.reapply_transforms = reapply_transforms
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._train_transform = self.get_train_transform()
        self._test_transform = self.get_test_transform()

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Dataset paradigm type. Must be one of DATASET_TYPES."""
        pass

    @abstractmethod
    def get_train_transform(self) -> transforms.Compose:
        """Get training data transforms."""
        pass

    @abstractmethod
    def get_test_transform(self) -> transforms.Compose:
        """Get testing data transforms."""
        pass

    @abstractmethod
    def load_data(self):
        """Load the dataset."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """Number of input channels."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Input image size (height, width)."""
        pass

    def get_dataloaders(
        self, batch_size: int = 64, num_workers: int = 4, shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test DataLoaders."""
        if self._train_dataset is None or self._test_dataset is None:
            self.load_data()

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def get_info(self) -> dict:
        """Get dataset information."""
        return {
            "name": self.__class__.__name__,
            "type": self.dataset_type,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "root": self.root,
        }

    def reset_train_transforms(self):
        """Reset training transforms to get new random augmentations.

        This should be called after each epoch when reapply_transforms is True
        to get different random augmentations for the training data.
        """
        if not self.reapply_transforms:
            return

        # Regenerate transforms (this creates new random state for random transforms)
        self._train_transform = self.get_train_transform()

        # Reload the training dataset with new transforms
        self._reload_train_data()

    @abstractmethod
    def _reload_train_data(self):
        """Reload training data with current transforms.

        Subclasses should implement this to reload the training dataset
        with the current self._train_transform.
        """
        pass
