"""
Event classifier — predicts usage/event suitability from clothing embeddings + gender.

Trained on wardrobe_v2 dataset.
Usage labels: Casual, Ethnic, Formal, Sports
Input: 2048-dim ResNet50 embedding + 3-dim gender one-hot = 2051 dims
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 4 usage classes (merged Smart Casual→Formal, Party→Casual, dropped Home/Travel)
USAGE_CLASSES = ["Casual", "Ethnic", "Formal", "Sports"]

# Gender classes (sorted for consistent one-hot encoding)
GENDER_CLASSES = ["Men", "Unisex", "Women"]

# Map user-facing event types to relevant usage classes with weights
EVENT_TO_USAGE = {
    "casual":     {"Casual": 1.0, "Sports": 0.3},
    "office":     {"Formal": 1.0, "Casual": 0.2},
    "wedding":    {"Ethnic": 1.0, "Formal": 0.7},
    "party":      {"Casual": 1.0, "Ethnic": 0.5},
    "date night": {"Formal": 0.8, "Casual": 0.6},
    "gym":        {"Sports": 1.0},
}

MODEL_PATH = "models/event_classifier.pth"
INPUT_DIM = 2051  # 2048 embedding + 3 gender one-hot


def encode_gender(gender: str) -> np.ndarray:
    """One-hot encode gender string. Returns 3-dim float32 array."""
    g = str(gender).strip().title()
    vec = np.zeros(len(GENDER_CLASSES), dtype=np.float32)
    if g in GENDER_CLASSES:
        vec[GENDER_CLASSES.index(g)] = 1.0
    else:
        vec[GENDER_CLASSES.index("Unisex")] = 1.0  # default for unknown
    return vec


def build_features(embeddings: np.ndarray, genders: list) -> np.ndarray:
    """
    Concatenate embeddings with gender one-hot vectors.

    Args:
        embeddings: (N, 2048) array
        genders:    list of N gender strings
    Returns:
        (N, 2051) array
    """
    gender_onehot = np.array([encode_gender(g) for g in genders], dtype=np.float32)
    return np.concatenate([embeddings, gender_onehot], axis=1)


class EventClassifierNet(nn.Module):
    """MLP that predicts usage probabilities from a (embedding + gender) feature vector."""

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)  # raw logits


def load_event_classifier(model_path: str = MODEL_PATH, device: str = "cpu",
                           input_dim: int = INPUT_DIM, num_classes: int = 4) -> EventClassifierNet:
    """Load a trained event classifier from checkpoint."""
    model = EventClassifierNet(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def predict_usage_batch(model: EventClassifierNet, embeddings: np.ndarray,
                         genders: list = None, device: str = "cpu") -> np.ndarray:
    """
    Predict usage probabilities for a batch of embeddings.

    Args:
        embeddings: (N, 2048) array
        genders:    list of N gender strings (defaults to 'Unisex' if None)
    Returns:
        (N, num_classes) array of probabilities
    """
    if genders is None:
        genders = ["Unisex"] * len(embeddings)
    features = build_features(embeddings, genders)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def compute_event_score(usage_probs: dict, event: str) -> float:
    """
    Compute how suitable an item is for a given event.

    Maps event type to relevant usage classes and computes a weighted score.
    Returns a score in [0, 1].
    """
    mapping = EVENT_TO_USAGE.get(event, EVENT_TO_USAGE["casual"])
    score = sum(usage_probs.get(cls, 0.0) * w for cls, w in mapping.items())
    max_score = sum(mapping.values())
    return min(score / max_score, 1.0)
