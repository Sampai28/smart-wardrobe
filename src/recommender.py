"""
Outfit recommender — generates top-N outfit combinations from the wardrobe.

Final score = compatibility_score * event_suitability_score
"""

import numpy as np
import itertools
from src.database import get_items
from src.compatibility import score_outfits_batch
from src.event_classifier import (
    EventClassifierNet, predict_usage_batch, compute_event_score
)


def get_top_outfits(conn, compat_weight: float = 0.5, event_weight: float = 0.5,
                    event: str = "casual", event_model: EventClassifierNet = None,
                    device: str = "cpu", top_n: int = 3) -> list:
    """
    Generate and rank all possible outfit combinations.

    Args:
        conn:           SQLite connection
        compat_weight:  Weight for compatibility score (0-1)
        event_weight:   Weight for event suitability score (0-1)
        event:          Event type string
        event_model:    Trained EventClassifierNet (optional — if None, event score = 1.0)
        device:         torch device
        top_n:          Number of top outfits to return

    Returns:
        List of dicts: [{top, bottom, shoes, scores}, ...]
    """
    tops = get_items(conn, "top")
    bottoms = get_items(conn, "bottom")
    shoes_items = get_items(conn, "shoes")

    if not tops or not bottoms or not shoes_items:
        return []

    # Generate all combinations
    combos = list(itertools.product(tops, bottoms, shoes_items))
    if not combos:
        return []

    # Stack embeddings for batch scoring
    top_embs = np.array([c[0]["embedding"] for c in combos])
    bottom_embs = np.array([c[1]["embedding"] for c in combos])
    shoes_embs = np.array([c[2]["embedding"] for c in combos])

    # Compatibility scores (batch)
    compat_scores = score_outfits_batch(top_embs, bottom_embs, shoes_embs)

    # Event suitability scores
    if event_model is not None:
        # Predict usage for all unique items (avoid duplicate inference)
        all_embs = np.concatenate([top_embs, bottom_embs, shoes_embs], axis=0)
        all_probs = predict_usage_batch(event_model, all_embs, device)

        n = len(combos)
        top_probs = all_probs[:n]
        bottom_probs = all_probs[n:2*n]
        shoes_probs = all_probs[2*n:]

        event_scores = np.zeros(n)
        for i in range(n):
            t_score = compute_event_score(
                {cls: float(p) for cls, p in zip(
                    ["Casual", "Ethnic", "Formal", "Home", "Party", "Smart Casual", "Sports", "Travel"],
                    top_probs[i])},
                event)
            b_score = compute_event_score(
                {cls: float(p) for cls, p in zip(
                    ["Casual", "Ethnic", "Formal", "Home", "Party", "Smart Casual", "Sports", "Travel"],
                    bottom_probs[i])},
                event)
            s_score = compute_event_score(
                {cls: float(p) for cls, p in zip(
                    ["Casual", "Ethnic", "Formal", "Home", "Party", "Smart Casual", "Sports", "Travel"],
                    shoes_probs[i])},
                event)
            event_scores[i] = (t_score + b_score + s_score) / 3
    else:
        event_scores = np.ones(len(combos))

    # Combined score
    final_scores = compat_weight * compat_scores + event_weight * event_scores

    # Rank and return top N
    top_indices = np.argsort(final_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        t, b, s = combos[idx]
        results.append({
            "top": {"id": t["id"], "name": t["name"], "thumbnail": t.get("thumbnail")},
            "bottom": {"id": b["id"], "name": b["name"], "thumbnail": b.get("thumbnail")},
            "shoes": {"id": s["id"], "name": s["name"], "thumbnail": s.get("thumbnail")},
            "scores": {
                "compatibility": round(float(compat_scores[idx]), 4),
                "event_suitability": round(float(event_scores[idx]), 4),
                "final": round(float(final_scores[idx]), 4),
            }
        })
    return results
