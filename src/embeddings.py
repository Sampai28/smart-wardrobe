"""
Feature extraction — ResNet50 (2048-dim) + CLIP ViT-B/32 (512-dim) = 2560-dim embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 2560  # 2048 (ResNet50) + 512 (CLIP)

# ResNet50 preprocessing
_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_feature_extractor():
    """Load ResNet50 (classification head removed) + CLIP ViT-B/32."""
    # ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_extractor = nn.Sequential(*list(resnet.children())[:-1])
    resnet_extractor.eval().to(DEVICE)

    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    return resnet_extractor, clip_model, clip_preprocess


def extract_embedding(extractor, image_path: str) -> np.ndarray:
    """Extract a 2560-dim embedding (ResNet50 + CLIP) from a single image."""
    resnet_extractor, clip_model, clip_preprocess = extractor
    img = Image.open(image_path).convert("RGB")

    # ResNet50 embedding (2048-dim)
    resnet_tensor = _resnet_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        resnet_emb = resnet_extractor(resnet_tensor).squeeze().cpu().numpy()

    # CLIP embedding (512-dim)
    clip_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        clip_emb = clip_model.encode_image(clip_tensor)
        clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        clip_emb = clip_emb.squeeze().cpu().float().numpy()

    return np.concatenate([resnet_emb, clip_emb])  # (2560,)


def extract_embeddings_batch(extractor, image_paths: list, batch_size: int = 64) -> np.ndarray:
    """Extract 2560-dim embeddings for a list of images in batches."""
    resnet_extractor, clip_model, clip_preprocess = extractor
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        resnet_tensors, clip_tensors = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                resnet_tensors.append(_resnet_transform(img))
                clip_tensors.append(clip_preprocess(img))
            except Exception:
                resnet_tensors.append(torch.zeros(3, 224, 224))
                clip_tensors.append(torch.zeros(3, 224, 224))

        # ResNet50 batch
        resnet_batch = torch.stack(resnet_tensors).to(DEVICE)
        with torch.no_grad():
            resnet_embs = resnet_extractor(resnet_batch).squeeze(-1).squeeze(-1).cpu().numpy()

        # CLIP batch
        clip_batch = torch.stack(clip_tensors).to(DEVICE)
        with torch.no_grad():
            clip_embs = clip_model.encode_image(clip_batch)
            clip_embs = clip_embs / clip_embs.norm(dim=-1, keepdim=True)
            clip_embs = clip_embs.cpu().float().numpy()

        combined = np.concatenate([resnet_embs, clip_embs], axis=1)  # (B, 2560)
        all_embeddings.append(combined)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images...")

    return np.concatenate(all_embeddings, axis=0)  # (N, 2560)
