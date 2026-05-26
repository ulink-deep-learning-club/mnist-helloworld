import torchvision
import torchvision.transforms as transforms
from .base import ClassificationDataset
from .utils import ResizePad


class CIFARDataset(ClassificationDataset):
    """CIFAR-10 dataset implementation."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 32,
        output_channels: int = 3,
    ):
        super().__init__(
            root, download, reapply_transforms, image_size, output_channels
        )

    def get_train_transform(
        self, image_size: int = 32, output_channels: int = 3
    ) -> transforms.Compose:
        # CIFAR images are natively RGB (3 channels).
        # Only apply Grayscale when explicitly converting to 1-channel.
        transform_list = []
        if output_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
            mean = (0.1307,)
            std = (0.3081,)
        else:
            # Use proper RGB normalization for native color images
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        transform_list.extend([
            ResizePad(image_size, pad_value=0),
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(transform_list)

    def get_test_transform(
        self, image_size: int = 32, output_channels: int = 3
    ) -> transforms.Compose:
        transform_list = []
        if output_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
            mean = (0.1307,)
            std = (0.3081,)
        else:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        transform_list.extend([
            ResizePad(image_size, pad_value=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(transform_list)

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
        return self.output_channels

    @property
    def input_size(self) -> tuple:
        return (self.image_size, self.image_size)
