# ============================================================
# Digital Wardrobe PoC — PyTorch Embeddings + Compatibility Score
# No API. No training. Fully local.
# Pipeline: Photos → ResNet50 embeddings → Compatibility score
# Requirements: pip install torch torchvision pillow scikit-learn
# ============================================================

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENTS = {
    "casual"    : [1, 0, 0, 0, 0, 0],
    "office"    : [0, 1, 0, 0, 0, 0],
    "wedding"   : [0, 0, 1, 0, 0, 0],
    "party"     : [0, 0, 0, 1, 0, 0],
    "date night": [0, 0, 0, 0, 1, 0],
    "gym"       : [0, 0, 0, 0, 0, 1],
}

# ── STEP 1: Load ResNet50 as Feature Extractor ────────────────
def load_feature_extractor():
    print(f"⏳ Loading ResNet50 on {DEVICE.upper()}...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove final classification layer → 2048-dim embeddings
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval().to(DEVICE)
    print("   ✅ ResNet50 ready")
    return extractor

# ── STEP 2: Preprocess Image ──────────────────────────────────
def preprocess(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

# ── STEP 3: Extract Embedding ─────────────────────────────────
def extract_embedding(extractor, image_path: str) -> np.ndarray:
    tensor = preprocess(image_path)
    with torch.no_grad():
        emb = extractor(tensor)
    return emb.squeeze().cpu().numpy()  # (2048,)

# ── STEP 4: Compatibility Score ───────────────────────────────
def compute_compatibility(top, bottom, shoes, event) -> dict:
    t = top.reshape(1, -1)
    b = bottom.reshape(1, -1)
    s = shoes.reshape(1, -1)

    sim_tb = float(cosine_similarity(t, b)[0][0])
    sim_ts = float(cosine_similarity(t, s)[0][0])
    sim_bs = float(cosine_similarity(b, s)[0][0])
    avg    = (sim_tb + sim_ts + sim_bs) / 3

    # Normalize [-1,1] → [0,1]
    score = (avg + 1) / 2

    # Event strictness weight
    weights = {
        "casual"    : 1.00,
        "gym"       : 1.10,
        "party"     : 0.95,
        "date night": 0.95,
        "office"    : 0.90,
        "wedding"   : 0.85,
    }
    score = min(score * weights.get(event, 1.0), 1.0)

    if score >= 0.75:   verdict = "Great Match  🟢"
    elif score >= 0.55: verdict = "Decent Match 🟡"
    else:               verdict = "Poor Match   🔴"

    return {
        "top_bottom_sim"   : round(sim_tb, 4),
        "top_shoes_sim"    : round(sim_ts, 4),
        "bottom_shoes_sim" : round(sim_bs, 4),
        "avg_similarity"   : round(avg, 4),
        "compatibility"    : round(score, 4),
        "verdict"          : verdict,
    }

# ── STEP 5: Print Report ──────────────────────────────────────
def print_report(scores, event, top_emb):
    sep = "=" * 54
    score = scores["compatibility"]
    bar   = "█" * int(score * 10) + "░" * (10 - int(score * 10))

    print(f"\n{sep}")
    print("  📊  EMBEDDING EXTRACTION")
    print(sep)
    print(f"  Shape  : {top_emb.shape}  (2048-dim vector per item)")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Sample (first 6 dims): {np.round(top_emb[:6], 3).tolist()}")

    print(f"\n{sep}")
    print(f"  👔  COMPATIBILITY — EVENT: {event.upper()}")
    print(sep)
    print(f"\n  Score   : [{bar}] {score:.0%}")
    print(f"  Verdict : {scores['verdict']}")
    print(f"\n  Pairwise Similarities:")
    print(f"    Top    ↔ Bottom : {scores['top_bottom_sim']}")
    print(f"    Top    ↔ Shoes  : {scores['top_shoes_sim']}")
    print(f"    Bottom ↔ Shoes  : {scores['bottom_shoes_sim']}")
    print(f"    Average         : {scores['avg_similarity']}")

    print(f"\n{sep}")
    print("  📌  NOTE: Cosine similarity is a PoC placeholder.")
    print("  Next: train a model on Polyvore for real fashion")
    print("  compatibility scoring.")
    print(sep)

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("\n👗  Digital Wardrobe — Embeddings PoC")
    print("    ResNet50 + Cosine Similarity (No API)")
    print("-" * 54)

    top_path    = input("Path to TOP image    : ").strip()
    bottom_path = input("Path to BOTTOM image : ").strip()
    shoes_path  = input("Path to SHOES image  : ").strip()

    for p in [top_path, bottom_path, shoes_path]:
        if not Path(p).exists():
            print(f"❌ File not found: {p}")
            return

    print(f"\nAvailable events : {', '.join(EVENTS.keys())}")
    event = input("Event type       : ").strip().lower()
    if event not in EVENTS:
        print(f"⚠️  Unknown event — defaulting to casual")
        event = "casual"

    # Load model
    extractor = load_feature_extractor()

    # Extract embeddings
    print("\n⏳ Extracting embeddings...")
    print("  [1/3] Top...")
    top_emb    = extract_embedding(extractor, top_path)
    print("  [2/3] Bottom...")
    bottom_emb = extract_embedding(extractor, bottom_path)
    print("  [3/3] Shoes...")
    shoes_emb  = extract_embedding(extractor, shoes_path)
    print("  ✅ Done")

    # Score
    print("\n⏳ Computing compatibility...")
    scores = compute_compatibility(top_emb, bottom_emb, shoes_emb, event)

    # Report
    print_report(scores, event, top_emb)

    # Save
    with open("poc_result.json", "w") as f:
        json.dump({
            "event" : event,
            "scores": scores,
            "embedding_shape": list(top_emb.shape)
        }, f, indent=2)
    print("\n💾 Results saved to poc_result.json\n")

if __name__ == "__main__":
    main()