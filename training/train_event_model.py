"""
Train the event classifier on the wardrobe_v2 dataset.

Usage:
    python -m training.train_event_model
"""

import os
import sys
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                              roc_auc_score, confusion_matrix, RocCurveDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import load_feature_extractor, extract_embeddings_batch, DEVICE
from src.event_classifier import EventClassifierNet, USAGE_CLASSES, INPUT_DIM, encode_gender, build_features

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "wardrobe_v2.zip"
EXTRACT_DIR = DATA_DIR / "wardrobe_v2"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache_resnet_clip.npz"
MODEL_SAVE_PATH = Path("models/event_classifier.pth")
METRICS_SAVE_PATH = Path("models/training_metrics.json")

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 7

CLASS_MERGE = {
    "Smart Casual": "Formal",
    "Party":        "Casual",
    "Home":         "Casual",
    "Travel":       "Casual",
}


class EmbeddingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def ensure_extracted() -> Path:
    if EXTRACT_DIR.exists() and (EXTRACT_DIR / "metadata.csv").exists():
        print(f"Dataset already extracted at {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print(f"  Extracted {len(list(EXTRACT_DIR.rglob('*')))} files")
    return EXTRACT_DIR


def load_data(extract_dir: Path) -> pd.DataFrame:
    print("Loading metadata.csv...")
    df = pd.read_csv(extract_dir / "metadata.csv")

    df["usage"] = df["usage"].map(lambda u: CLASS_MERGE.get(u, u))
    df = df.dropna(subset=["usage", "gender"]).copy()
    df = df[df["usage"].isin(USAGE_CLASSES)].copy()

    def find_image(row):
        for subdir in ["tops", "bottoms", "shoes"]:
            p = extract_dir / subdir / row["file_name"]
            if p.exists():
                return str(p)
        return None

    df["image_path"] = df.apply(find_image, axis=1)
    df = df[df["image_path"].notna()].copy()

    label_map = {cls: i for i, cls in enumerate(USAGE_CLASSES)}
    df["label"] = df["usage"].map(label_map)

    print(f"  Valid items: {len(df)}")
    for cls in USAGE_CLASSES:
        count = len(df[df["usage"] == cls])
        print(f"    {cls}: {count} ({count / len(df) * 100:.1f}%)")

    return df


def get_embeddings(df: pd.DataFrame) -> np.ndarray:
    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        data = np.load(EMBEDDINGS_CACHE)
        cached_ids = data["ids"]
        cached_embs = data["embeddings"]

        df_ids = df["id"].values
        cached_id_set = set(cached_ids.tolist())
        if all(int(i) in cached_id_set for i in df_ids):
            id_to_idx = {int(cid): idx for idx, cid in enumerate(cached_ids)}
            indices = [id_to_idx[int(i)] for i in df_ids]
            print(f"  Cache hit: {len(indices)} embeddings")
            return cached_embs[indices]
        else:
            print("  Cache mismatch, recomputing...")

    print(f"Extracting CLIP embeddings for {len(df)} images on {DEVICE}...")
    extractor = load_feature_extractor()
    image_paths = df["image_path"].tolist()
    embeddings = extract_embeddings_batch(extractor, image_paths, batch_size=64)

    np.savez(EMBEDDINGS_CACHE, ids=df["id"].values, embeddings=embeddings)
    print(f"  Cached {len(embeddings)} embeddings to {EMBEDDINGS_CACHE}")
    return embeddings


def train():
    extract_dir = ensure_extracted()
    df = load_data(extract_dir)
    embeddings = get_embeddings(df)

    genders = df["gender"].tolist()
    features = build_features(embeddings, genders)
    labels = df["label"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    num_classes = len(USAGE_CLASSES)
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = DEVICE
    model = EventClassifierNet(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"Training on {device.upper() if isinstance(device, str) else str(device).upper()}...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Acc':>10}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(feats)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                logits = model(feats)
                val_loss += criterion(logits, lbls).item() * len(feats)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_true.extend(lbls.cpu().numpy())
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

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

    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()

    def evaluate(loader, split_name):
        preds, trues, probs_all = [], [], []
        with torch.no_grad():
            for feats, lbls in loader:
                feats = feats.to(device)
                logits = model(feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_all.extend(probs)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(lbls.numpy())
        return np.array(trues), np.array(preds), np.array(probs_all)

    val_true, val_preds, val_probs   = evaluate(val_loader,  "Val")
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=BATCH_SIZE)
    test_true, test_preds, test_probs = evaluate(test_loader, "Test")

    num_cls = len(USAGE_CLASSES)
    present_labels = sorted(set(test_true) | set(test_preds))
    present_names  = [USAGE_CLASSES[i] for i in present_labels]

    print("\n=== Validation Classification Report ===")
    print(classification_report(val_true, val_preds,
                                labels=present_labels, target_names=present_names, zero_division=0))

    print("=== Test Classification Report ===")
    print(classification_report(test_true, test_preds,
                                labels=present_labels, target_names=present_names, zero_division=0))

    # AUC-ROC (one-vs-rest, macro)
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(test_true, classes=list(range(num_cls)))
    auc_macro = roc_auc_score(y_test_bin, test_probs, average="macro", multi_class="ovr")
    print(f"Test AUC-ROC (macro OvR): {auc_macro:.4f}")

    plots_dir = MODEL_SAVE_PATH.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss"); axes[0].legend()
    axes[1].plot(epochs_range, [v * 100 for v in history["val_acc"]], label="Val Acc")
    axes[1].axhline(y=80, color="r", linestyle="--", label="80% target")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()

    cm = confusion_matrix(test_true, test_preds, labels=list(range(num_cls)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_cls)); ax.set_yticks(range(num_cls))
    ax.set_xticklabels(USAGE_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(USAGE_CLASSES)
    for i in range(num_cls):
        for j in range(num_cls):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "darkorange", "green", "red"]
    for i, (cls_name, color) in enumerate(zip(USAGE_CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - One-vs-Rest (Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {plots_dir}/")

    # Save metrics
    metrics = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(accuracy_score(test_true, test_preds)),
        "test_auc_roc_macro": float(auc_macro),
        "epochs_trained": len(history["train_loss"]),
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "history": history,
        "val_classification_report": classification_report(
            val_true, val_preds, labels=present_labels,
            target_names=present_names, output_dict=True, zero_division=0
        ),
        "test_classification_report": classification_report(
            test_true, test_preds, labels=present_labels,
            target_names=present_names, output_dict=True, zero_division=0
        ),
    }
    METRICS_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    train()
