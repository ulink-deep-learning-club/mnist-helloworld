import os
import numpy as np
import torch
import torchvision
from torch.utils.data import random_split, Subset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .base import BaseDataset


class AlbumentationsTransform:
    """Wrapper to use Albumentations transforms with torchvision datasets."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        # Apply Albumentations transform with named argument
        transformed = self.transform(image=img_np)
        return transformed["image"]


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
                AlbumentationsTransform(
                    A.Compose(
                        [
                            # 几何变换：模拟手写自然倾斜、缩放
                            A.Affine(
                                scale=(0.8, 1.2),  # 随机缩放 ±20%
                                translate_percent=(-0.1, 0.1),  # 随机平移 ±10%
                                rotate=(-15, 15),  # 随机旋转 ±15°
                                shear=(-10, 10),  # 随机剪切 ±10°
                                p=0.8,
                            ),
                            # 弹性变形：模拟手写笔画的自然弯曲
                            A.ElasticTransform(
                                alpha=1,
                                sigma=50,
                                p=0.3,
                            ),
                            # 笔画粗细变化：使用形态学腐蚀/膨胀模拟
                            A.RandomGridShuffle(
                                grid=(2, 2), p=0.2
                            ),  # 随机打乱局部块（强制模型学习结构）
                            # 局部遮挡：模拟污渍或笔画缺失
                            A.CoarseDropout(
                                num_holes_range=(1, 4),  # 孔洞数量范围
                                hole_height_range=(
                                    0.05,
                                    0.1,
                                ),  # 孔洞高度占图像高度的比例
                                hole_width_range=(
                                    0.05,
                                    0.1,
                                ),  # 孔洞宽度占图像宽度的比例
                                fill=0,  # 填充值
                                p=0.3,
                            ),
                            # 对比度/亮度变化：模拟不同纸张/扫描仪
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.2),
                                contrast_limit=(-0.2, 0.2),
                                p=0.5,
                            ),
                            # 高斯噪声：模拟传感器噪声
                            A.GaussNoise(p=0.2),
                            # 最终处理
                            A.Resize(64, 64),
                            A.Normalize(mean=0.5, std=0.5),  # 如果输入是灰度图，可简化
                            ToTensorV2(),
                        ]
                    )
                ),
            ]
        )

    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                AlbumentationsTransform(
                    A.Compose(
                        [
                            # 几何变换：模拟手写自然倾斜、缩放
                            A.Affine(
                                scale=(0.8, 1.2),  # 随机缩放 ±20%
                                translate_percent=(-0.1, 0.1),  # 随机平移 ±10%
                                rotate=(-15, 15),  # 随机旋转 ±15°
                                shear=(-10, 10),  # 随机剪切 ±10°
                                p=0.8,
                            ),
                            # 弹性变形：模拟手写笔画的自然弯曲
                            A.ElasticTransform(
                                alpha=1,
                                sigma=50,
                                p=0.3,
                            ),
                            # 笔画粗细变化：使用形态学腐蚀/膨胀模拟
                            A.RandomGridShuffle(
                                grid=(2, 2), p=0.2
                            ),  # 随机打乱局部块（强制模型学习结构）
                            # 局部遮挡：模拟污渍或笔画缺失
                            A.CoarseDropout(
                                num_holes_range=(1, 4),  # 孔洞数量范围
                                hole_height_range=(
                                    0.05,
                                    0.1,
                                ),  # 孔洞高度占图像高度的比例
                                hole_width_range=(
                                    0.05,
                                    0.1,
                                ),  # 孔洞宽度占图像宽度的比例
                                fill=0,  # 填充值
                                p=0.3,
                            ),
                            # 对比度/亮度变化：模拟不同纸张/扫描仪
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.2),
                                contrast_limit=(-0.2, 0.2),
                                p=0.5,
                            ),
                            # 高斯噪声：模拟传感器噪声
                            A.GaussNoise(p=0.2),
                            # 最终处理
                            A.Resize(64, 64),
                            A.Normalize(mean=0.5, std=0.5),  # 如果输入是灰度图，可简化
                            ToTensorV2(),
                        ]
                    )
                ),
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
