# 👗 Digital Wardrobe — Outfit Recommender

A machine learning system that learns your wardrobe and recommends the **top 3 outfit combinations** for any event — fully local, no external APIs.

---

## 🧠 How It Works

```
Upload clothing photo (one at a time)
        ↓
ResNet50 extracts 2048-dim embedding
(original image discarded)
        ↓
Embedding + metadata stored in SQLite
        ↓
User selects event type
        ↓
All wardrobe combinations scored by
compatibility model
        ↓
Top 3 outfits returned
```

---

## 📁 Project Structure

```
digital-wardrobe/
├── main.py                  # Embedding extraction + compatibility scoring
├── wardrobe.db              # SQLite wardrobe store (auto-created)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (if needed)
├── .gitignore
├── README.md                # This file
├── sample_clothes/          # Test images
│   ├── top.jpg
│   ├── bottom.jpg
│   └── shoes.jpg
└── poc_result.json          # PoC output (auto-generated)
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/digital-wardrobe.git
cd digital-wardrobe
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU (optional but recommended)
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

---

## ▶️ Run

```bash
python main.py
```

When prompted:
```
Path to TOP image    : sample_clothes/top.jpg
Path to BOTTOM image : sample_clothes/bottom.jpg
Path to SHOES image  : sample_clothes/shoes.jpg

Available events: casual, office, wedding, party, date night, gym
Event type: casual
```

---

## 📊 Sample Output

```
👗  Digital Wardrobe — Embeddings PoC
    ResNet50 + Cosine Similarity (No API)
------------------------------------------------------
⏳ Loading ResNet50 on CUDA...
   ✅ ResNet50 ready

⏳ Extracting embeddings...
  [1/3] Top...
  [2/3] Bottom...
  [3/3] Shoes...
  ✅ Done

======================================================
  📊  EMBEDDING EXTRACTION
======================================================
  Shape  : (2048,)  (2048-dim vector per item)
  Device : CUDA

======================================================
  👔  COMPATIBILITY — EVENT: CASUAL
======================================================
  Score   : [████████░░] 80%
  Verdict : Great Match 🟢

  Pairwise Similarities:
    Top    ↔ Bottom : 0.4821
    Top    ↔ Shoes  : 0.3102
    Bottom ↔ Shoes  : 0.3754
    Average         : 0.3892
======================================================
💾 Results saved to poc_result.json
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch 2.9.1 | Model training and embedding extraction |
| torchvision | Pretrained ResNet50 feature extractor |
| scikit-learn | Cosine similarity baseline + evaluation metrics |
| SQLite | Persistent wardrobe store (embeddings + metadata) |
| Pillow | Image loading and preprocessing |
| Streamlit | Frontend UI (coming in v2) |

---

## 🗺️ Roadmap

- [x] ResNet50 embedding extraction (CUDA)
- [x] Cosine similarity baseline scoring
- [x] End-to-end PoC pipeline
- [ ] SQLite wardrobe store (append embeddings as items are added)
- [ ] Train compatibility model on Polyvore dataset
- [ ] Top 3 outfit ranking from full wardrobe
- [ ] Streamlit UI — upload, manage wardrobe, view recommendations
- [ ] Final evaluation — Accuracy, F1, AUC-ROC

---

## 📦 Requirements

```
torch==2.9.1
torchvision==0.24.1
pillow
scikit-learn
python-dotenv
```

Install with:
```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install pillow scikit-learn python-dotenv
```

---

## ⚠️ Current Limitations (PoC Stage)

- Compatibility scoring uses cosine similarity — a placeholder until the model is trained on Polyvore
- No persistent wardrobe storage yet — each run is independent
- Works best with well-lit, single-item photos against a plain background

---

## 🤖 AI Acknowledgement

This project was designed iteratively with the assistance of Claude (Anthropic) for idea refinement, code structure, and documentation. The original concept involved using Gemini Flash Lite for embedding extraction — through iterative refinement, the project evolved into a fully local ML pipeline using ResNet50. All AI assistance is explicitly acknowledged here and in the project report.

---

## 📄 License

MIT