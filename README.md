# Smart Wardrobe

A locally deployed, privacy-preserving outfit recommendation system. Upload photos of your clothing, pick an event and gender filter, and get ranked outfit combinations from your own wardrobe. No cloud, no external APIs.

---

## How It Works

1. **Upload** clothing photos (tops, bottoms, shoes) with a name, category, and gender tag
2. Each image is embedded using **ResNet50** (2048-dim) + **CLIP ViT-B/32** (512-dim) > 2560-dim hybrid vector
3. Embeddings + a gender one-hot (3-dim) > **2563-dim input** to an MLP event classifier
4. On the Recommend page, all top x bottom x shoes combinations are enumerated, gender-filtered, and scored
5. Outfits are ranked by **event confidence** (how well each item suits the selected occasion)

---

## Features

- Hybrid ResNet50 + CLIP embeddings for richer visual + semantic features
- Gender-aware recommendations (Men / Women / Unisex filter)
- 6 event types: Casual, Office, Wedding, Party, Date Night, Gym
- Automatic embedding migration, re-embeds stored items if the backbone changes, no re-upload needed
- Fully local, no internet connection required at inference time
- Privacy-preserving, original images discarded after embedding extraction

---

## Event Classifier

Trained on **wardrobe_v2**, a balanced subset of 6,000 items (2,000 tops/bottoms/shoes) from the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

| Class | Support |
|---|---|
| Casual | 68.3% |
| Sports | 16.2% |
| Formal | 8.3% |
| Ethnic | 7.1% |

**Results (70/15/15 stratified split)**

| Metric | Value |
|---|---|
| Test Accuracy | 88% |
| Macro AUC-ROC | 0.9649 |
| Weighted F1 | 0.88 |

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/Sampai28/smart-wardrobe.git
cd smart-wardrobe
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Prepare the dataset

Place the `wardrobe_v2.zip` file in the `data/` folder

```
data/
  wardrobe_v2.zip
```

### 3. Train the event classifier

```bash
python -m training.train_event_model
```

This will
- Extract `wardrobe_v2.zip` to `data/wardrobe_v2/`
- Compute and cache embeddings to `data/embeddings_cache_hybrid.npz`
- Train the MLP for up to 40 epochs with early stopping
- Save the best model to `models/event_classifier_hybrid.pth`
- Save plots (training curves, confusion matrix, ROC curves) to `models/plots/`

### 4. Run the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
app/
  app.py                            Streamlit UI (Upload, My Wardrobe, Recommend)
src/
  embeddings.py                     ResNet50 + CLIP hybrid feature extraction
  database.py                       SQLite wardrobe store + embedding migration gate
  compatibility.py                  Cosine similarity compatibility scoring
  event_classifier.py               MLP model, gender encoding, event-to-usage mapping
  recommender.py                    Combination enumeration, gender filtering, ranking
training/
  train_event_model.py              Full training pipeline (supports --backbone flag)
models/
  event_classifier_hybrid.pth       Trained MLP weights (hybrid backbone)
  training_metrics_hybrid.json      Accuracy, AUC-ROC, classification report
  plots/                            Training curves, confusion matrix, ROC curves
data/
  wardrobe_v2.zip                   Training dataset
  embeddings_cache_hybrid.npz       Cached embeddings (auto-generated)
report/
  literature_survey.tex             IEEE conference paper
```

---

## Tech Stack

| Component | Library |
|---|---|
| Visual embeddings | PyTorch, torchvision (ResNet50) |
| Semantic embeddings | OpenAI CLIP (ViT-B/32) |
| Event classifier | PyTorch MLP |
| Wardrobe storage | SQLite |
| UI | Streamlit |
| Metrics | scikit-learn |
| Data processing | NumPy, Pandas |

---

## Acknowledgement

Built with assistance from [Claude](https://claude.ai) (Anthropic) for system architecture and documentation.
