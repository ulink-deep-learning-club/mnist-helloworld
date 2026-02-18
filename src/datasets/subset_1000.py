import os
import numpy as np
import random
from tqdm import tqdm
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

    @property
    def dataset_type(self) -> str:
        return "standard"

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


class TripletSubset1000Dataset(BaseDataset):
    """Subset 1000 dataset for triplet learning."""

    @property
    def dataset_type(self) -> str:
        return "triplet"

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        triplets_per_class: int = 100,
    ):
        super().__init__(root, download, reapply_transforms)
        self.triplets_per_class = triplets_per_class

    def get_train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                AlbumentationsTransform(
                    A.Compose(
                        [
                            A.Affine(
                                scale=(0.8, 1.2),
                                translate_percent=(-0.1, 0.1),
                                rotate=(-15, 15),
                                shear=(-10, 10),
                                p=0.8,
                            ),
                            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                            A.RandomGridShuffle(grid=(2, 2), p=0.2),
                            A.CoarseDropout(
                                num_holes_range=(1, 4),
                                hole_height_range=(0.05, 0.1),
                                hole_width_range=(0.05, 0.1),
                                fill=0,
                                p=0.3,
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.2),
                                contrast_limit=(-0.2, 0.2),
                                p=0.5,
                            ),
                            A.GaussNoise(p=0.2),
                            A.Resize(64, 64),
                            A.Normalize(mean=0.5, std=0.5),
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
                            A.Resize(64, 64),
                            A.Normalize(mean=0.5, std=0.5),
                            ToTensorV2(),
                        ]
                    )
                ),
            ]
        )

    def load_data(self):
        """Load Subset 1000 dataset and generate triplets."""
        data_path = os.path.join(self.root, "subset_1000")

        # Load full dataset without transform first to organize by label
        print("Loading dataset from disk...")
        full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=None)

        # Organize by label with progress bar
        print(f"Organizing {len(full_dataset)} samples by label...")
        data_by_label = {}
        # Use imgs attribute directly for faster access (avoids __getitem__ overhead)
        for idx, (_, label) in enumerate(full_dataset.imgs):
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append(idx)

        # Split into train/test (80/20)
        train_indices = []
        test_indices = []
        for label, indices in data_by_label.items():
            train_size = int(0.8 * len(indices))
            train_indices.extend(indices[:train_size])
            test_indices.extend(indices[train_size:])

        # Create datasets with transforms
        from .triplet_mnist import FixedTripletDataset

        print(f"Generating triplets for {len(data_by_label)} classes...")
        train_triplets = self._generate_triplets(
            full_dataset,
            data_by_label,
            train_indices,
            self.triplets_per_class,
            desc="Train triplets",
        )
        test_triplets = self._generate_triplets(
            full_dataset,
            data_by_label,
            test_indices,
            self.triplets_per_class // 10,
            desc="Test triplets",
        )

        self._train_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=train_triplets,
            transform=self._train_transform,
        )
        self._test_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=test_triplets,
            transform=self._test_transform,
        )

        # Store train indices for reload
        train_set = set(train_indices)

        print("Filtering train indices by label...")
        self._train_indices_by_label = {
            label: [idx for idx in indices if idx in train_set]
            for label, indices in data_by_label.items()
        }

    def _generate_triplets(
        self, base_dataset, data_by_label, available_indices, per_class, desc="Triplets"
    ):
        """Generate balanced triplets from available indices."""

        triplets = []

        # Convert to set for O(1) lookup - critical for performance
        available_set = set(available_indices)

        # Filter data_by_label to only include available indices
        available_by_label = {}
        for label, indices in data_by_label.items():
            available = [idx for idx in indices if idx in available_set]
            if len(available) >= 2:  # Need at least 2 for anchor and positive
                available_by_label[label] = available

        labels = list(available_by_label.keys())
        total_triplets = len(labels) * per_class

        with tqdm(total=total_triplets, desc=desc, unit=" triplet") as pbar:
            for anchor_label in labels:
                anchor_indices = available_by_label[anchor_label]
                for _ in range(per_class):
                    if len(anchor_indices) < 2:
                        pbar.update(1)
                        continue
                    # Sample anchor and positive
                    anchor_idx = random.choice(anchor_indices)
                    positive_idx = random.choice(anchor_indices)
                    while positive_idx == anchor_idx:
                        positive_idx = random.choice(anchor_indices)

                    # Sample negative from different label
                    negative_label = random.choice(
                        [label for label in labels if label != anchor_label]
                    )
                    negative_idx = random.choice(available_by_label[negative_label])

                    triplets.append(
                        (anchor_idx, positive_idx, negative_idx, anchor_label)
                    )
                    pbar.update(1)

        return triplets

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        data_path = os.path.join(self.root, "subset_1000")
        full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=None)

        from .triplet_mnist import FixedTripletDataset

        print("Regenerating training triplets...")
        train_triplets = self._generate_triplets(
            full_dataset,
            self._train_indices_by_label,
            [
                idx
                for indices in self._train_indices_by_label.values()
                for idx in indices
            ],
            self.triplets_per_class,
            desc="Regen triplets",
        )

        self._train_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=train_triplets,
            transform=self._train_transform,
        )

    @property
    def num_classes(self) -> int:
        return 1000

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
