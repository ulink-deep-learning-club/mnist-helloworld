import torchvision
import torchvision.transforms as transforms
from .base import BaseDataset


class CIFARDataset(BaseDataset):
    """CIFAR-10 dataset implementation."""

    @property
    def dataset_type(self) -> str:
        return "standard"

    def get_train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def load_data(self):
        """Load CIFAR-10 dataset."""
        self._train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=self.download,
            transform=self._train_transform,
        )
        self._test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=self.download,
            transform=self._test_transform,
        )

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        self._train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=False, transform=self._train_transform
        )

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def input_size(self) -> tuple:
        return (32, 32)
