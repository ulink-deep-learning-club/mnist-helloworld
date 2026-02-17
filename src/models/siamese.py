import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from .base import BaseModel
from ..training.metrics import TripletMetricsTracker


class SiameseNetwork(BaseModel):
    """
    Siamese Neural Network with triplet loss support.

    Uses a shared backbone to extract embeddings from input images.
    Suitable for metric learning and similarity comparison tasks.
    """

    @property
    def model_type(self) -> str:
        return "siamese"

    @classmethod
    def get_criterion(cls, margin: float = 1.0, **kwargs) -> nn.Module:
        """Return TripletLoss for siamese network."""
        return TripletLoss(margin=margin)

    @classmethod
    def get_metrics_tracker(cls, margin: float = 1.0, **kwargs) -> Any:
        """Return triplet metrics tracker."""
        return TripletMetricsTracker(margin=margin)

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        input_size: tuple = (28, 28),
        embedding_dim: int = 128,
        **kwargs,
    ):
        super().__init__(num_classes, input_channels)
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        # Shared backbone - feature extractor
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Calculate the flattened size
        h, w = input_size
        for _ in range(3):  # 3 maxpool layers with stride 2
            h = h // 2
            w = w // 2
        self.flattened_size = 128 * h * w

        # Embedding layer - maps to embedding space
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
        )

        # Optional classifier head (for when you want to convert to classification)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            If in training mode: L2-normalized embeddings of shape (B, embedding_dim)
            If in eval mode: Raw embeddings (not normalized)
        """
        features = self.backbone(x)
        embedding = self.embedding(features)

        # L2 normalize embeddings during training for triplet loss
        if self.training:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_with_classifier(self, x):
        """
        Forward pass with classification head.
        Useful when you want to use the learned embeddings for classification.
        """
        embedding = self.forward(x)
        return self.classifier(embedding)

    def get_embedding(self, x):
        """Get raw embedding without normalization."""
        features = self.backbone(x)
        return self.embedding(features)


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.

    L(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
    where d(x, y) is the Euclidean distance between embeddings.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Embeddings of anchor samples (B, embedding_dim)
            positive: Embeddings of positive samples (B, embedding_dim)
            negative: Embeddings of negative samples (B, embedding_dim)

        Returns:
            Triplet loss value
        """
        # Euclidean distance
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)

        return losses.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Mining Loss.

    Instead of pre-selected triplets, selects hard triplets from a batch.
    This is more efficient and produces better results.
    """

    def __init__(self, margin: float = 1.0, soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Ground truth labels (B,)

        Returns:
            Triplet loss value and number of valid triplets
        """
        # Compute pairwise distance matrix
        distance_matrix = self._pairwise_distances(embeddings)

        # Get hardest positive and negative for each anchor
        triplets = self._get_triplets(labels, distance_matrix)

        if len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device), 0

        # Extract distances
        anchor_idx = triplets[:, 0]
        positive_idx = triplets[:, 1]
        negative_idx = triplets[:, 2]

        pos_dist = distance_matrix[anchor_idx, positive_idx]
        neg_dist = distance_matrix[anchor_idx, negative_idx]

        if self.soft_margin:
            # Soft margin: log(1 + exp(d(a,p) - d(a,n)))
            losses = F.softplus(pos_dist - neg_dist)
        else:
            # Hard margin: max(d(a,p) - d(a,n) + margin, 0)
            losses = F.relu(pos_dist - neg_dist + self.margin)

        return losses.mean(), len(triplets)

    def _pairwise_distances(self, embeddings):
        """Compute pairwise L2 distance matrix."""
        # embeddings: (B, D)
        dot_product = torch.matmul(embeddings, embeddings.t())  # (B, B)

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        # Since embeddings are L2 normalized, ||a||^2 = 1
        squared_norm = torch.sum(embeddings**2, dim=1, keepdim=True)  # (B, 1)
        distances = squared_norm + squared_norm.t() - 2.0 * dot_product
        distances = F.relu(distances)  # Ensure non-negative

        return torch.sqrt(distances + 1e-8)

    def _get_triplets(self, labels, distance_matrix):
        """Mine hard triplets from the batch."""
        triplets = []

        for i in range(len(labels)):
            anchor_label = labels[i]

            # Find positives (same label as anchor)
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
            positive_indices = torch.where(positive_mask)[0]

            # Find negatives (different label from anchor)
            negative_mask = labels != anchor_label
            negative_indices = torch.where(negative_mask)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            # Hardest positive: positive with maximum distance
            hardest_positive = positive_indices[
                torch.argmax(distance_matrix[i, positive_indices])
            ]

            # Hardest negative: negative with minimum distance
            hardest_negative = negative_indices[
                torch.argmin(distance_matrix[i, negative_indices])
            ]

            triplets.append([i, hardest_positive.item(), hardest_negative.item()])

        return torch.tensor(triplets, device=labels.device)


if __name__ == "__main__":
    # Test the model
    model = SiameseNetwork(input_channels=1, input_size=(28, 28), embedding_dim=128)

    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm (should be ~1): {torch.norm(embedding, dim=1)}")

    # Test triplet loss
    anchor = torch.randn(4, 128)
    positive = torch.randn(4, 128)
    negative = torch.randn(4, 128)

    criterion = TripletLoss(margin=1.0)
    loss = criterion(anchor, positive, negative)
    print(f"Triplet loss: {loss.item()}")
