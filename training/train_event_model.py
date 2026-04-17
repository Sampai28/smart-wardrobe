"""
Train the event classifier on the Fashion Product Images dataset.

Usage:
    python -m training.train_event_model

Pipeline:
    1. Load styles.csv, filter to items with valid usage + images
    2. Pre-compute ResNet50 embeddings (cached to disk)
    3. Train MLP classifier: embedding (2048) -> usage class (8)
    4. Save best checkpoint to models/event_classifier.pth
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import load_feature_extractor, extract_embeddings_batch, DEVICE
from src.event_classifier import EventClassifierNet, USAGE_CLASSES

# ── CONFIG ────────────────────────────────────────────────────
DATA_DIR = Path("data")
STYLES_CSV = DATA_DIR / "styles.csv"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache.npz"
MODEL_SAVE_PATH = Path("models/event_classifier.pth")
METRICS_SAVE_PATH = Path("models/training_metrics.json")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5  # early stopping


# ── DATASET ───────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ── STEP 1: LOAD & FILTER DATA ───────────────────────────────
def load_data() -> pd.DataFrame:
    print("Loading styles.csv...")
    df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")

    # Keep only rows with valid usage labels
    df = df[df["usage"].isin(USAGE_CLASSES)].copy()

    # Build image paths and check existence
    df["image_path"] = df["id"].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
    df["image_exists"] = df["image_path"].apply(os.path.exists)
    df = df[df["image_exists"]].copy()

    # Drop classes with too few samples (need at least 2 for stratified split)
    class_counts = df["usage"].value_counts()
    small_classes = class_counts[class_counts < 5].index.tolist()
    if small_classes:
        print(f"  Dropping tiny classes (<5 samples): {small_classes}")
        df = df[~df["usage"].isin(small_classes)].copy()

    # Encode labels
    label_map = {cls: i for i, cls in enumerate(USAGE_CLASSES)}
    df["label"] = df["usage"].map(label_map)

    print(f"  Valid items: {len(df)}")
    print(f"  Class distribution:")
    for cls in USAGE_CLASSES:
        count = len(df[df["usage"] == cls])
        print(f"    {cls:15s}: {count}")

    return df


# ── STEP 2: PRE-COMPUTE EMBEDDINGS ───────────────────────────
def get_embeddings(df: pd.DataFrame) -> np.ndarray:
    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        data = np.load(EMBEDDINGS_CACHE)
        cached_ids = data["ids"]
        cached_embs = data["embeddings"]

        # Check if all needed IDs are in the cache
        df_ids = df["id"].values
        cached_id_set = set(cached_ids.tolist())
        if all(i in cached_id_set for i in df_ids):
            # Build index map and extract matching embeddings in order
            id_to_idx = {int(cid): idx for idx, cid in enumerate(cached_ids)}
            indices = [id_to_idx[int(i)] for i in df_ids]
            print(f"  Cache hit: using {len(indices)} of {len(cached_embs)} cached embeddings")
            return cached_embs[indices]
        else:
            print("  Cache mismatch — recomputing...")

    print(f"Extracting ResNet50 embeddings for {len(df)} images...")
    print(f"  Device: {DEVICE}")
    print(f"  This may take a while on CPU...")

    extractor = load_feature_extractor()
    image_paths = df["image_path"].tolist()
    embeddings = extract_embeddings_batch(extractor, image_paths, batch_size=64)

    # Cache to disk
    np.savez(EMBEDDINGS_CACHE, ids=df["id"].values, embeddings=embeddings)
    print(f"  Cached {len(embeddings)} embeddings to {EMBEDDINGS_CACHE}")

    return embeddings


# ── STEP 3: TRAIN ─────────────────────────────────────────────
def train():
    df = load_data()
    embeddings = get_embeddings(df)
    labels = df["label"].values

    # Train/val split (80/20, stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

    # Weighted sampler to handle class imbalance
    class_counts = np.bincount(y_train, minlength=len(USAGE_CLASSES))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    device = DEVICE
    model = EventClassifierNet(num_classes=len(USAGE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining on {device.upper()}...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 44)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for embs, lbls in train_loader:
            embs, lbls = embs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(embs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(embs)
        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for embs, lbls in val_loader:
                embs, lbls = embs.to(device), lbls.to(device)
                logits = model(embs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * len(embs)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_true.extend(lbls.cpu().numpy())
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            marker = " *"
        else:
            patience_counter += 1

        print(f"{epoch+1:>6} {train_loss:>12.4f} {val_loss:>12.4f} {val_acc:>9.2%}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Final evaluation
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Load best model and print classification report
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for embs, lbls in val_loader:
            embs, lbls = embs.to(device), lbls.to(device)
            logits = model(embs)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_true.extend(lbls.cpu().numpy())

    # Only include class names that actually appear in predictions
    present_labels = sorted(set(val_true) | set(val_preds))
    present_names = [USAGE_CLASSES[i] for i in present_labels]

    print("\nClassification Report:")
    print(classification_report(val_true, val_preds, labels=present_labels,
                                target_names=present_names, zero_division=0))

    # Save metrics
    metrics = {
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
        "classification_report": classification_report(
            val_true, val_preds, labels=present_labels,
            target_names=present_names, output_dict=True, zero_division=0
        ),
    }
    METRICS_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    train()
