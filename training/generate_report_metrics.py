"""
Compute and save evaluation metrics for the report.

Run with: python -m training.generate_report_metrics
Outputs go to report-metrics/
"""

import sys
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_classifier import EventClassifierNet, USAGE_CLASSES, build_features

DATA_DIR     = Path("data")
ZIP_PATH     = DATA_DIR / "wardrobe_v2.zip"
EXTRACT_DIR  = DATA_DIR / "wardrobe_v2"
HYBRID_CACHE = DATA_DIR / "embeddings_cache_hybrid.npz"
METRICS_JSON = Path("models/training_metrics_hybrid.json")
OUT_DIR      = Path("report-metrics")

INPUT_DIM   = 2563
NUM_CLASSES = len(USAGE_CLASSES)
BATCH_SIZE  = 64
EPOCHS      = 40
PATIENCE    = 7
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MERGE = {
    "Smart Casual": "Formal",
    "Party":        "Casual",
    "Home":         "Casual",
    "Travel":       "Casual",
}


class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data():
    if not EXTRACT_DIR.exists() or not (EXTRACT_DIR / "metadata.csv").exists():
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

    df = pd.read_csv(EXTRACT_DIR / "metadata.csv")
    df["usage"] = df["usage"].map(lambda u: CLASS_MERGE.get(u, u))
    df = df.dropna(subset=["usage", "gender"]).copy()
    df = df[df["usage"].isin(USAGE_CLASSES)].copy()

    def find_image(row):
        for subdir in ["tops", "bottoms", "shoes"]:
            p = EXTRACT_DIR / subdir / row["file_name"]
            if p.exists():
                return str(p)
        return None

    df["image_path"] = df.apply(find_image, axis=1)
    df = df[df["image_path"].notna()].copy()
    df["label"] = df["usage"].map({cls: i for i, cls in enumerate(USAGE_CLASSES)})
    return df


def load_features(df):
    print(f"Loading hybrid embeddings from {HYBRID_CACHE}...")
    cache = np.load(HYBRID_CACHE)
    id_to_idx = {int(cid): idx for idx, cid in enumerate(cache["ids"])}

    missing = [i for i in df["id"].values if int(i) not in id_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} item IDs not found in cache.")

    indices    = [id_to_idx[int(i)] for i in df["id"].values]
    embeddings = cache["embeddings"][indices]
    features   = build_features(embeddings, df["gender"].tolist())
    labels     = df["label"].values
    print(f"  Features: {features.shape}")
    return features, labels


def train_fold(X_train, y_train, X_val, y_val):
    class_counts   = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights  = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds   = EmbeddingDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model     = EventClassifierNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc     = 0.0
    best_state       = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for feats, lbls in train_dl:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(feats), lbls).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for feats, lbls in val_dl:
                feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
                logits = model(feats)
                val_loss += criterion(logits, lbls).item() * len(feats)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(lbls.cpu().numpy())
        val_loss /= len(val_ds)
        val_acc   = accuracy_score(trues, preds)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for feats, lbls in val_dl:
            feats  = feats.to(DEVICE)
            logits = model(feats)
            probs  = torch.softmax(logits, 1).cpu().numpy()
            probs_all.extend(probs)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())

    return best_val_acc, np.array(trues), np.array(preds), np.array(probs_all)


def run_kfold(features, labels, n_splits=5):
    print(f"\nRunning {n_splits}-fold stratified cross-validation...")
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"  Fold {fold_idx + 1}/{n_splits}...", end=" ", flush=True)
        X_tr, X_vl = features[train_idx], features[val_idx]
        y_tr, y_vl = labels[train_idx], labels[val_idx]

        best_acc, y_true, y_pred, y_prob = train_fold(X_tr, y_tr, X_vl, y_vl)

        report = classification_report(
            y_true, y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=USAGE_CLASSES,
            output_dict=True, zero_division=0
        )
        y_bin     = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")

        folds.append({
            "fold":          fold_idx + 1,
            "val_accuracy":  round(float(best_acc), 4),
            "auc_roc_macro": round(float(auc_macro), 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"],    4),
                    "f1":        round(report[cls]["f1-score"],  4),
                    "support":   int(report[cls]["support"]),
                } for cls in USAGE_CLASSES
            },
        })
        print(f"acc={best_acc:.4f}  auc={auc_macro:.4f}")

    accs = [f["val_accuracy"]  for f in folds]
    aucs = [f["auc_roc_macro"] for f in folds]
    print(f"\nCV Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"CV AUC-ROC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return {
        "n_splits":      n_splits,
        "folds":         folds,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy":  round(float(np.std(accs)),  4),
        "mean_auc_roc":  round(float(np.mean(aucs)), 4),
        "std_auc_roc":   round(float(np.std(aucs)),  4),
    }


def build_summary_metrics(metrics_json_path):
    with open(metrics_json_path) as f:
        m = json.load(f)

    test_cr   = m["test_classification_report"]
    per_class = {}
    for cls in USAGE_CLASSES:
        cr = test_cr[cls]
        per_class[cls] = {
            "accuracy":  round(cr["recall"],    4),
            "precision": round(cr["precision"], 4),
            "recall":    round(cr["recall"],    4),
            "f1":        round(cr["f1-score"],  4),
            "support":   int(cr["support"]),
        }

    return {
        "test_accuracy":      round(m["test_acc"], 4),
        "best_val_accuracy":  round(m["best_val_acc"], 4),
        "test_auc_roc_macro": round(m["test_auc_roc_macro"], 4),
        "epochs_trained":     m["epochs_trained"],
        "dataset_split": {
            "train": m["split"]["train"],
            "val":   m["split"]["val"],
            "test":  m["split"]["test"],
            "total": m["split"]["train"] + m["split"]["val"] + m["split"]["test"],
        },
        "macro_avg": {
            "precision": round(test_cr["macro avg"]["precision"], 4),
            "recall":    round(test_cr["macro avg"]["recall"],    4),
            "f1":        round(test_cr["macro avg"]["f1-score"],  4),
        },
        "weighted_avg": {
            "precision": round(test_cr["weighted avg"]["precision"], 4),
            "recall":    round(test_cr["weighted avg"]["recall"],    4),
            "f1":        round(test_cr["weighted avg"]["f1-score"],  4),
        },
        "per_class": per_class,
    }


def build_overfitting_analysis(metrics_json_path):
    with open(metrics_json_path) as f:
        m = json.load(f)

    hist    = m["history"]
    n       = len(hist["train_loss"])
    rows    = []
    for i in range(n):
        rows.append({
            "epoch":      i + 1,
            "train_loss": round(hist["train_loss"][i], 4),
            "val_loss":   round(hist["val_loss"][i],   4),
            "val_acc":    round(hist["val_acc"][i],    4),
            "loss_gap":   round(hist["val_loss"][i] - hist["train_loss"][i], 4),
        })

    best_epoch       = int(np.argmin(hist["val_loss"])) + 1
    final_train_loss = hist["train_loss"][-1]
    final_val_loss   = hist["val_loss"][-1]

    return {
        "epochs_trained":          n,
        "best_val_loss_epoch":     best_epoch,
        "best_val_accuracy":       round(max(hist["val_acc"]), 4),
        "final_train_loss":        round(final_train_loss, 4),
        "final_val_loss":          round(final_val_loss,   4),
        "overfit_gap":             round(final_val_loss - final_train_loss, 4),
        "early_stopping_patience": PATIENCE,
        "epoch_history":           rows,
    }


def evaluate_test_split(features, labels):
    print("\nEvaluating on test split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model_path = Path("models/event_classifier_hybrid.pth")
    if not model_path.exists():
        model_path = Path("models/event_classifier.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = EventClassifierNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval().to(DEVICE)

    test_dl = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=BATCH_SIZE)
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for feats, lbls in test_dl:
            feats  = feats.to(DEVICE)
            logits = model(feats)
            probs  = torch.softmax(logits, 1).cpu().numpy()
            probs_all.extend(probs)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())

    return np.array(trues), np.array(preds), np.array(probs_all)


def plot_training_curves(metrics_json_path, out_dir):
    with open(metrics_json_path) as f:
        m = json.load(f)

    hist   = m["history"]
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, hist["val_loss"],   label="Val Loss",   color="darkorange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    val_acc_pct = [v * 100 for v in hist["val_acc"]]
    axes[1].plot(epochs, val_acc_pct, label="Val Accuracy", color="green")
    axes[1].axhline(y=max(val_acc_pct), color="red", linestyle="--",
                    label=f"Best: {max(val_acc_pct):.1f}%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy over Epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / 'training_curves.png'}")


def plot_confusion_matrix(y_true, y_pred, out_dir):
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(USAGE_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(USAGE_CLASSES)
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / 'confusion_matrix.png'}")


def plot_roc_curves(y_true, y_prob, out_dir):
    y_bin  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ["steelblue", "darkorange", "green", "red"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cls_name, color) in enumerate(zip(USAGE_CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - One-vs-Rest (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / 'roc_curves.png'}")


def plot_kfold_results(kfold_data, out_dir):
    folds    = [f["fold"]         for f in kfold_data["folds"]]
    accs     = [f["val_accuracy"] for f in kfold_data["folds"]]
    mean_acc = kfold_data["mean_accuracy"]
    std_acc  = kfold_data["std_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(folds, [a * 100 for a in accs], color="steelblue", alpha=0.8, label="Fold Accuracy")
    ax.axhline(y=mean_acc * 100, color="red", linestyle="--",
               label=f"Mean: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}%")
    ax.fill_between(
        [0.5, len(folds) + 0.5],
        [(mean_acc - std_acc) * 100] * 2,
        [(mean_acc + std_acc) * 100] * 2,
        color="red", alpha=0.1, label="Std Dev Band"
    )
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"{kfold_data['n_splits']}-Fold Stratified Cross-Validation")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kfold_accuracy_plot.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / 'kfold_accuracy_plot.png'}")


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    features, labels = load_features(df)

    summary = build_summary_metrics(METRICS_JSON)
    save_json(summary, OUT_DIR / "summary_metrics.json")

    overfit = build_overfitting_analysis(METRICS_JSON)
    save_json(overfit, OUT_DIR / "overfitting_analysis.json")

    per_class_acc = {
        cls: {
            "per_class_accuracy": summary["per_class"][cls]["accuracy"],
            "support":            summary["per_class"][cls]["support"],
        }
        for cls in USAGE_CLASSES
    }
    save_json(per_class_acc, OUT_DIR / "per_class_accuracy.json")

    y_true, y_pred, y_prob = evaluate_test_split(features, labels)
    plot_training_curves(METRICS_JSON, OUT_DIR)
    plot_confusion_matrix(y_true, y_pred, OUT_DIR)
    plot_roc_curves(y_true, y_prob, OUT_DIR)

    kfold_data = run_kfold(features, labels, n_splits=5)
    save_json(kfold_data, OUT_DIR / "kfold_results.json")
    plot_kfold_results(kfold_data, OUT_DIR)

    print("\nTest Accuracy:     ", summary["test_accuracy"])
    print("Test AUC-ROC:      ", summary["test_auc_roc_macro"])
    print("Best Val Accuracy: ", summary["best_val_accuracy"])
    print("CV Accuracy:       ", f"{kfold_data['mean_accuracy']} +/- {kfold_data['std_accuracy']}")
    print("CV AUC-ROC:        ", f"{kfold_data['mean_auc_roc']} +/- {kfold_data['std_auc_roc']}")


if __name__ == "__main__":
    main()
