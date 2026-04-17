"""
Smart Wardrobe — CLI entry point

Demonstrates the full pipeline: embedding extraction, compatibility scoring,
and event-aware outfit recommendation.
"""

import json
import numpy as np
from pathlib import Path

from src.embeddings import load_feature_extractor, extract_embedding, DEVICE
from src.compatibility import score_outfit
from src.event_classifier import (
    load_event_classifier, compute_outfit_event_score,
    EVENT_TO_USAGE, MODEL_PATH
)


def main():
    print("\n👗  Smart Wardrobe — Outfit Scorer")
    print(f"    ResNet50 + Event Classifier (Device: {DEVICE.upper()})")
    print("-" * 54)

    # Get image paths
    top_path = input("Path to TOP image    : ").strip()
    bottom_path = input("Path to BOTTOM image : ").strip()
    shoes_path = input("Path to SHOES image  : ").strip()

    for p in [top_path, bottom_path, shoes_path]:
        if not Path(p).exists():
            print(f"  File not found: {p}")
            return

    # Event selection
    events = list(EVENT_TO_USAGE.keys())
    print(f"\nAvailable events : {', '.join(events)}")
    event = input("Event type       : ").strip().lower()
    if event not in events:
        print(f"  Unknown event — defaulting to casual")
        event = "casual"

    # Load models
    extractor = load_feature_extractor()

    event_model = None
    if Path(MODEL_PATH).exists():
        print("Loading event classifier...")
        event_model = load_event_classifier(MODEL_PATH, device=DEVICE)
        print("  Event classifier ready")
    else:
        print("  No event classifier found — using compatibility only")
        print(f"  Train one with: python -m training.train_event_model")

    # Extract embeddings
    print("\nExtracting embeddings...")
    top_emb = extract_embedding(extractor, top_path)
    bottom_emb = extract_embedding(extractor, bottom_path)
    shoes_emb = extract_embedding(extractor, shoes_path)
    print("  Done")

    # Compatibility score
    scores = score_outfit(top_emb, bottom_emb, shoes_emb)

    # Event score
    event_score = None
    if event_model:
        event_score = compute_outfit_event_score(
            event_model, top_emb, bottom_emb, shoes_emb, event, DEVICE
        )

    # Combined score
    if event_score is not None:
        final = 0.5 * scores["compatibility_score"] + 0.5 * event_score
    else:
        final = scores["compatibility_score"]

    # Verdict
    if final >= 0.75:
        verdict = "Great Match  🟢"
    elif final >= 0.55:
        verdict = "Decent Match 🟡"
    else:
        verdict = "Poor Match   🔴"

    # Report
    sep = "=" * 54
    bar = "█" * int(final * 10) + "░" * (10 - int(final * 10))

    print(f"\n{sep}")
    print(f"  📊  EMBEDDING EXTRACTION")
    print(sep)
    print(f"  Shape  : {top_emb.shape}  (2048-dim vector per item)")
    print(f"  Device : {DEVICE.upper()}")

    print(f"\n{sep}")
    print(f"  👔  OUTFIT SCORE — EVENT: {event.upper()}")
    print(sep)
    print(f"\n  Final Score : [{bar}] {final:.0%}")
    print(f"  Verdict     : {verdict}")
    print(f"\n  Compatibility : {scores['compatibility_score']:.2%}")
    if event_score is not None:
        print(f"  Event Match   : {event_score:.2%}")
    print(f"\n  Pairwise Similarities:")
    print(f"    Top    ↔ Bottom : {scores['top_bottom_sim']}")
    print(f"    Top    ↔ Shoes  : {scores['top_shoes_sim']}")
    print(f"    Bottom ↔ Shoes  : {scores['bottom_shoes_sim']}")
    print(f"    Average         : {scores['avg_similarity']}")
    print(sep)

    # Save result
    result = {
        "event": event,
        "compatibility": scores,
        "event_suitability": event_score,
        "final_score": round(final, 4),
        "verdict": verdict.strip(),
    }
    with open("poc_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Results saved to poc_result.json\n")


if __name__ == "__main__":
    main()
