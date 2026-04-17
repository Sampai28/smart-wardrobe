"""
ResNet50 feature extraction — extracts 2048-dim embeddings from clothing images.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_feature_extractor():
    """Load ResNet50 with final classification layer removed."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval().to(DEVICE)
    return extractor


def preprocess(image_path: str) -> torch.Tensor:
    """Load an image and return a preprocessed tensor."""
    img = Image.open(image_path).convert("RGB")
    return _transform(img).unsqueeze(0).to(DEVICE)


def extract_embedding(extractor, image_path: str) -> np.ndarray:
    """Extract a 2048-dim embedding from a single image."""
    tensor = preprocess(image_path)
    with torch.no_grad():
        emb = extractor(tensor)
    return emb.squeeze().cpu().numpy()  # (2048,)


def extract_embeddings_batch(extractor, image_paths: list, batch_size: int = 32) -> np.ndarray:
    """Extract embeddings for a list of images in batches."""
    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                tensors.append(preprocess(p).squeeze(0))
            except Exception:
                # Skip broken images, insert zeros
                tensors.append(torch.zeros(3, 224, 224, device=DEVICE))
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            embs = extractor(batch)
        all_embeddings.append(embs.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)  # (N, 2048)
