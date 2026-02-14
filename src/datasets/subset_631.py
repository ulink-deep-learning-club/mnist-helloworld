import torchvision
import torchvision.transforms as transforms
from .base import BaseDataset


class Subset631Dataset(BaseDataset):
    """Subset 631 Chinese character dataset."""

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
        """Load Subset 631 dataset."""
        import os

        data_path = os.path.join(self.root, "subset_631")

        self._train_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )
        self._test_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._test_transform
        )

    @property
    def num_classes(self) -> int:
        return 631

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def input_size(self) -> tuple:
        return (64, 64)
