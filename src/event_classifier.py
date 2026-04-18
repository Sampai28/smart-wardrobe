"""
Event classifier - predicts usage/event suitability from clothing embeddings + gender.
Trained on wardrobe_v2. Labels: Casual, Ethnic, Formal, Sports.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

USAGE_CLASSES = ["Casual", "Ethnic", "Formal", "Sports"]
GENDER_CLASSES = ["Men", "Unisex", "Women"]

EVENT_TO_USAGE = {
    "casual":     {"Casual": 1.0, "Sports": 0.3},
    "office":     {"Formal": 1.0, "Casual": 0.2},
    "wedding":    {"Ethnic": 1.0, "Formal": 0.7},
    "party":      {"Casual": 1.0, "Ethnic": 0.5},
    "date night": {"Formal": 0.8, "Casual": 0.6},
    "gym":        {"Sports": 1.0},
}

MODEL_PATH = "models/event_classifier.pth"
INPUT_DIM = 2563


def encode_gender(gender: str) -> np.ndarray:
    """One-hot encode a gender string. Returns a 3-dim float32 array."""
    g = str(gender).strip().title()
    vec = np.zeros(len(GENDER_CLASSES), dtype=np.float32)
    if g in GENDER_CLASSES:
        vec[GENDER_CLASSES.index(g)] = 1.0
    else:
        vec[GENDER_CLASSES.index("Unisex")] = 1.0
    return vec


def build_features(embeddings: np.ndarray, genders: list) -> np.ndarray:
    """Concatenate embeddings with gender one-hot vectors."""
    gender_onehot = np.array([encode_gender(g) for g in genders], dtype=np.float32)
    return np.concatenate([embeddings, gender_onehot], axis=1)


class EventClassifierNet(nn.Module):
    """MLP that predicts usage probabilities from embedding + gender features."""

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
        return self.network(x)


def load_event_classifier(model_path: str = MODEL_PATH, device: str = "cpu",
                           input_dim: int = INPUT_DIM, num_classes: int = 4) -> EventClassifierNet:
    """Load a trained event classifier from a checkpoint file."""
    model = EventClassifierNet(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def predict_usage_batch(model: EventClassifierNet, embeddings: np.ndarray,
                         genders: list = None, device: str = "cpu") -> np.ndarray:
    """Predict usage class probabilities for a batch of embeddings."""
    if genders is None:
        genders = ["Unisex"] * len(embeddings)
    features = build_features(embeddings, genders)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def compute_event_score(usage_probs: dict, event: str) -> float:
    """Compute how suitable an item is for a given event. Returns a score in [0, 1]."""
    mapping = EVENT_TO_USAGE.get(event, EVENT_TO_USAGE["casual"])
    score = sum(usage_probs.get(cls, 0.0) * w for cls, w in mapping.items())
    max_score = sum(mapping.values())
    return min(score / max_score, 1.0)
