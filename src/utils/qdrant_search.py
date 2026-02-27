"""
Qdrant Vector Search Utility for Model Embeddings

This module provides utilities to:
1. Extract embeddings from a dataset using a trained model
2. Store embeddings in Qdrant vector database
3. Search for similar images using vector similarity

Usage:
    # Index embeddings from a checkpoint
    python -m src.utils.qdrant_search index \
        --checkpoint runs/exp1/checkpoints/best_model.pt \
        --dataset subset_1000 \
        --collection chinese_characters

    # Search for similar images
    python -m src.utils.qdrant_search search \
        --checkpoint runs/exp1/checkpoints/best_model.pt \
        --collection chinese_characters \
        --image path/to/query/image.png
"""

import argparse
from typing import List, Optional, Tuple
from pathlib import Path
from torchvision import transforms

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import our framework modules
try:
    from ..models import ModelRegistry
    from ..datasets import DatasetRegistry
    from ..datasets.base import BaseDataset
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models import ModelRegistry
    from src.datasets import DatasetRegistry
    from src.datasets.base import BaseDataset


class QdrantEmbeddingIndexer:
    """Index model embeddings in Qdrant for similarity search."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 256,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = vector_size
        self.distance = distance

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
            print(f"Collection created with vector size: {self.vector_size}")
        else:
            print(f"Using existing collection: {self.collection_name}")

    def index_embeddings(
        self,
        embeddings: np.ndarray,
        labels: List[int],
        ids: Optional[List[int]] = None,
        payloads: Optional[List[dict]] = None,
        batch_size: int = 100,
    ):
        """
        Index embeddings in Qdrant.

        Args:
            embeddings: Array of shape (N, vector_size)
            labels: List of class labels
            ids: Optional list of point IDs (default: auto-generated)
            payloads: Optional list of payload dicts with metadata
            batch_size: Number of points to insert per batch
        """
        num_points = len(embeddings)
        print(f"Indexing {num_points} embeddings...")

        points = []
        for i in range(num_points):
            point_id = ids[i] if ids else i
            payload = payloads[i] if payloads else {"label": int(labels[i])}

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )

        # Insert in batches
        for i in tqdm(range(0, len(points), batch_size), desc="Indexing"):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        print(f"Successfully indexed {num_points} embeddings")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_label: Optional[int] = None,
    ) -> List[dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_label: Optional label to filter results

        Returns:
            List of search results with id, score, and payload
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if label specified
        query_filter = None
        if filter_label is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="label",
                        match=MatchValue(value=filter_label),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        ).points

        print(results)

        return [
            {
                "id": result.id,
                "score": result.score,
                "label": result.payload.get("label") if result.payload else "",
                "payload": result.payload,
            }
            for result in results
        ]

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu", embedding_dim: int = 256
):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_name = model_config.get("name", "siamese_fpn_vit")

    # Map class names to registry names (case-insensitive matching)
    class_name_to_registry = {
        "SiameseFPNViT": "siamese_fpn_vit",
        "SiameseFPNViTTiny": "siamese_fpn_vit_tiny",
        "SiameseFPNViTSmall": "siamese_fpn_vit_small",
        "SiameseFPNViTLarge": "siamese_fpn_vit_large",
        "SiameseNetwork": "siamese",
    }

    # Try to map class name to registry name
    if model_name in class_name_to_registry:
        registry_name = class_name_to_registry[model_name]
        print(f"Mapped '{model_name}' to registry name '{registry_name}'")
        model_name = registry_name

    # Use CLI embedding_dim if provided, otherwise from checkpoint, otherwise default
    if embedding_dim is None:
        embedding_dim = model_config.get("embedding_dim", 256)
    print(f"Using embedding_dim: {embedding_dim}")

    # Create model
    model = ModelRegistry.create(
        model_name,
        num_classes=model_config.get("num_classes", 1000),
        input_channels=model_config.get("input_channels", 3),
        input_size=model_config.get("input_size", (64, 64)),
        embedding_dim=embedding_dim,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded: {model_name}")
    return model


def extract_embeddings(
    model: torch.nn.Module,
    dataset: BaseDataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from dataset.

    Returns:
        embeddings: Array of shape (N, embedding_dim)
        labels: Array of shape (N,)
    """

    # Get dataloader
    _, val_loader = dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=4,
        shuffle_train=False,
    )

    embeddings_list = []
    labels_list = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing"):
            # Handle both standard and triplet datasets
            if dataset.dataset_type == "triplet":
                # Triplet dataset returns (anchor, positive, negative, label)
                images = batch[0]  # Use anchor
                labels = batch[3]
            else:
                images, labels = batch

            images = images.to(device)

            # Get embeddings
            emb = model(images)

            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(
                labels.numpy() if isinstance(labels, torch.Tensor) else labels
            )

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)

    print(
        f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )
    return embeddings, labels


def get_device(force_cpu: bool = False) -> str:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def index_command(args):
    """Index embeddings from a checkpoint."""
    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint, device, embedding_dim=args.embedding_dim
    )

    # Get embedding dimension from model
    embedding_dim = getattr(model, "embedding_dim", args.embedding_dim)

    # Create dataset
    dataset = DatasetRegistry.create(
        args.dataset,
        root=args.data_root,
        download=False,
    )

    # Extract embeddings
    embeddings, labels = extract_embeddings(
        model, dataset, device, batch_size=args.batch_size
    )

    # Create indexer and index embeddings
    indexer = QdrantEmbeddingIndexer(
        collection_name=args.collection,
        host=args.qdrant_host,
        port=args.qdrant_port,
        vector_size=embedding_dim,
        distance=Distance.COSINE,
    )

    # Prepare payloads with image paths and character text if available
    payloads = None
    test_dataset = dataset._test_dataset

    assert isinstance(test_dataset, torch.utils.data.Dataset), "Test dataset is None"

    # Handle Subset wrapper (from random_split) to access underlying dataset
    base_dataset = test_dataset
    if hasattr(test_dataset, "dataset"):
        base_dataset = test_dataset.dataset # pyright: ignore[reportAttributeAccessIssue]

    if hasattr(base_dataset, "imgs") and hasattr(base_dataset, "classes"):
        # Get the indices if it's a Subset
        indices = None
        if hasattr(test_dataset, "indices"):
            indices = test_dataset.indices # pyright: ignore[reportAttributeAccessIssue]

        payloads = []
        for i, label in enumerate(labels):
            # Get the original index in the full dataset
            if indices is not None:
                orig_idx = indices[i]
            else:
                orig_idx = i

            # Get image path and character
            image_path = base_dataset.imgs[orig_idx][0] # pyright: ignore[reportAttributeAccessIssue]
            character = base_dataset.classes[label] # pyright: ignore[reportAttributeAccessIssue]

            payloads.append(
                {
                    "label": int(label),
                    "image_path": image_path,
                    "character": character,
                }
            )

    indexer.index_embeddings(
        embeddings=embeddings,
        labels=labels.tolist(),
        payloads=payloads,
        batch_size=args.batch_size,
    )

    # Print collection info
    info = indexer.get_collection_info()
    print(f"Collection info: {info}")


def search_command(args):
    """Search for similar images."""
    device = get_device(force_cpu=args.cpu)

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint, device, embedding_dim=args.embedding_dim
    )

    # Create indexer
    embedding_dim = getattr(model, "embedding_dim", 256)
    indexer = QdrantEmbeddingIndexer(
        collection_name=args.collection,
        host=args.qdrant_host,
        port=args.qdrant_port,
        vector_size=embedding_dim,
    )

    # Load and preprocess query image
    print(f"Loading query image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    # Default transform
    default_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = default_transform(image).unsqueeze(0).to(device) # pyright: ignore[reportAttributeAccessIssue]
    min_val = image_tensor.min()
    image_tensor = (image_tensor - min_val) / (image_tensor.max() - min_val) * 255
    # save tensor to image
    from torchvision.utils import save_image

    save_image(image_tensor, "/Users/anson/Projects/mnist-helloworld/test_image.png")
    print("Image saved as test_image.png")

    # Extract embedding
    with torch.no_grad():
        query_embedding = model(image_tensor).cpu().numpy()[0]

    # Search
    print(f"Searching for top {args.top_k} similar images...")
    results = indexer.search(
        query_vector=query_embedding,
        top_k=args.top_k,
        filter_label=args.filter_label,
    )

    # Display results
    print("\nSearch Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        character = result.get("payload", {}).get("character", "")
        char_display = f" (Character: {character})" if character else ""
        print(
            f"{i}. ID: {result['id']}, Score: {result['score']:.4f}, Label: {result['label']}{char_display}"
        )
        if "image_path" in result.get("payload", {}):
            print(f"   Path: {result['payload']['image_path']}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Vector Search for Model Embeddings"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index embeddings from checkpoint"
    )
    index_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    index_parser.add_argument(
        "--dataset",
        type=str,
        default="subset_1000",
        help="Dataset name",
    )
    index_parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection name",
    )
    index_parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root directory",
    )
    index_parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant server host",
    )
    index_parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for extraction",
    )
    index_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension (must match checkpoint)",
    )
    index_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of CUDA",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar images")
    search_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    search_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to query image",
    )
    search_parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection name",
    )
    search_parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (for transforms)",
    )
    search_parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root directory",
    )
    search_parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant server host",
    )
    search_parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    search_parser.add_argument(
        "--filter-label",
        type=int,
        help="Filter results by label",
    )
    search_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension (must match checkpoint)",
    )
    search_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of CUDA",
    )

    args = parser.parse_args()

    if args.command == "index":
        index_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
