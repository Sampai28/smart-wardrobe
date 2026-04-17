"""
Event classifier — predicts usage/event suitability from clothing embeddings.

Trained on the Fashion Product Images dataset (styles.csv).
Usage labels: Casual, Sports, Ethnic, Formal, Smart Casual, Party, Travel, Home
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# The 8 usage classes from styles.csv (sorted for consistency)
USAGE_CLASSES = ["Casual", "Ethnic", "Formal", "Home", "Party", "Smart Casual", "Sports", "Travel"]

# Map user-facing event types to relevant usage classes with weights
EVENT_TO_USAGE = {
    "casual":     {"Casual": 1.0, "Smart Casual": 0.7, "Travel": 0.5},
    "office":     {"Formal": 1.0, "Smart Casual": 0.8},
    "wedding":    {"Ethnic": 1.0, "Formal": 0.8, "Party": 0.6},
    "party":      {"Party": 1.0, "Smart Casual": 0.7, "Casual": 0.3},
    "date night": {"Smart Casual": 1.0, "Party": 0.8, "Casual": 0.4},
    "gym":        {"Sports": 1.0},
}

MODEL_PATH = "models/event_classifier.pth"


class EventClassifierNet(nn.Module):
    """MLP that predicts usage probabilities from a 2048-dim ResNet50 embedding."""

    def __init__(self, embedding_dim: int = 2048, num_classes: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
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


def load_event_classifier(model_path: str = MODEL_PATH, device: str = "cpu") -> EventClassifierNet:
    """Load a trained event classifier from checkpoint."""
    model = EventClassifierNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def predict_usage(model: EventClassifierNet, embedding: np.ndarray, device: str = "cpu") -> dict:
    """
    Predict usage probabilities for a single item embedding.

    Returns dict mapping usage class names to probabilities.
    """
    tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return {cls: float(prob) for cls, prob in zip(USAGE_CLASSES, probs)}


def predict_usage_batch(model: EventClassifierNet, embeddings: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Predict usage probabilities for a batch of embeddings.

    Args:
        embeddings: (N, 2048) array
    Returns:
        (N, 8) array of probabilities
    """
    tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
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
    score = 0.0
    for usage_class, weight in mapping.items():
        score += usage_probs.get(usage_class, 0.0) * weight
    # Normalize by max possible weight sum
    max_score = sum(mapping.values())
    return min(score / max_score, 1.0)


def compute_outfit_event_score(model: EventClassifierNet, top_emb: np.ndarray,
                                bottom_emb: np.ndarray, shoes_emb: np.ndarray,
                                event: str, device: str = "cpu") -> float:
    """
    Compute the event suitability score for a complete outfit.
    Average of individual item event scores.
    """
    top_probs = predict_usage(model, top_emb, device)
    bottom_probs = predict_usage(model, bottom_emb, device)
    shoes_probs = predict_usage(model, shoes_emb, device)

    top_score = compute_event_score(top_probs, event)
    bottom_score = compute_event_score(bottom_probs, event)
    shoes_score = compute_event_score(shoes_probs, event)

    return (top_score + bottom_score + shoes_score) / 3
