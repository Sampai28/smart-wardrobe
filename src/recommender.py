"""
Outfit recommender — generates top-N outfit combinations from the wardrobe.

Final score = compat_weight * compatibility + event_weight * event_suitability
"""

import numpy as np
import itertools
from src.database import get_items
from src.compatibility import score_outfits_batch
from src.event_classifier import (
    EventClassifierNet, USAGE_CLASSES, predict_usage_batch, compute_event_score
)


def _filter_by_gender(items: list, gender_filter: str) -> list:
    """
    Keep items that match the selected gender or are Unisex.

    gender_filter: "Men", "Women", or "All" (no filtering)
    """
    if gender_filter == "All":
        return items
    return [item for item in items if item.get("gender", "Unisex") in (gender_filter, "Unisex")]


def get_top_outfits(conn, compat_weight: float = 0.5, event_weight: float = 0.5,
                    event: str = "casual", event_model: EventClassifierNet = None,
                    device: str = "cpu", top_n: int = 3,
                    gender_filter: str = "All") -> list:
    """
    Generate and rank all possible outfit combinations.

    Args:
        conn:           SQLite connection
        compat_weight:  Weight for compatibility score (0-1)
        event_weight:   Weight for event suitability score (0-1)
        event:          Event type string
        event_model:    Trained EventClassifierNet (optional — if None, event score = 1.0)
        device:         torch device string
        top_n:          Number of top outfits to return
        gender_filter:  "Men", "Women", or "All" — filters items before combining

    Returns:
        List of dicts: [{top, bottom, shoes, scores}, ...]
    """
    tops        = _filter_by_gender(get_items(conn, "top"),    gender_filter)
    bottoms     = _filter_by_gender(get_items(conn, "bottom"), gender_filter)
    shoes_items = _filter_by_gender(get_items(conn, "shoes"),  gender_filter)

    if not tops or not bottoms or not shoes_items:
        return []

    combos = list(itertools.product(tops, bottoms, shoes_items))
    if not combos:
        return []

    # Stack embeddings and genders for batch scoring
    top_embs    = np.array([c[0]["embedding"] for c in combos])
    bottom_embs = np.array([c[1]["embedding"] for c in combos])
    shoes_embs  = np.array([c[2]["embedding"] for c in combos])

    top_genders    = [c[0].get("gender", "Unisex") for c in combos]
    bottom_genders = [c[1].get("gender", "Unisex") for c in combos]
    shoes_genders  = [c[2].get("gender", "Unisex") for c in combos]

    # Compatibility scores (batch, no gender needed)
    compat_scores = score_outfits_batch(top_embs, bottom_embs, shoes_embs)

    # Event suitability scores
    if event_model is not None:
        n = len(combos)
        top_probs    = predict_usage_batch(event_model, top_embs,    top_genders,    device)
        bottom_probs = predict_usage_batch(event_model, bottom_embs, bottom_genders, device)
        shoes_probs  = predict_usage_batch(event_model, shoes_embs,  shoes_genders,  device)

        event_scores = np.zeros(n)
        for i in range(n):
            t_score = compute_event_score(
                {cls: float(p) for cls, p in zip(USAGE_CLASSES, top_probs[i])}, event)
            b_score = compute_event_score(
                {cls: float(p) for cls, p in zip(USAGE_CLASSES, bottom_probs[i])}, event)
            s_score = compute_event_score(
                {cls: float(p) for cls, p in zip(USAGE_CLASSES, shoes_probs[i])}, event)
            event_scores[i] = (t_score + b_score + s_score) / 3
    else:
        event_scores = np.ones(len(combos))

    # Combined score (kept for reference)
    final_scores = compat_weight * compat_scores + event_weight * event_scores

    # Rank by event score if model available, otherwise by compatibility
    ranking_scores = event_scores if event_model is not None else compat_scores
    top_indices = np.argsort(ranking_scores, kind="stable")[::-1][:top_n]

    results = []
    for idx in top_indices:
        t, b, s = combos[idx]
        results.append({
            "top":    {"id": t["id"], "name": t["name"], "thumbnail": t.get("thumbnail")},
            "bottom": {"id": b["id"], "name": b["name"], "thumbnail": b.get("thumbnail")},
            "shoes":  {"id": s["id"], "name": s["name"], "thumbnail": s.get("thumbnail")},
            "scores": {
                "compatibility":   round(float(compat_scores[idx]), 4),
                "event_suitability": round(float(event_scores[idx]), 4),
                "final":           round(float(final_scores[idx]), 4),
            }
        })
    return results
