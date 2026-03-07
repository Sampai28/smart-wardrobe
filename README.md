<<<<<<< HEAD
# 👗 Digital Wardrobe — Outfit Recommender

A machine learning system that learns your wardrobe and recommends the **top 3 outfit combinations** for any event — fully local, no external APIs.
=======
# smart-wardrobe# 👗 Digital Wardrobe — Outfit Recommendation PoC

A Python proof-of-concept that classifies clothing items from photos and recommends outfit combinations based on the event type — using Google Gemini Vision.
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde

---

## 🧠 How It Works

```
<<<<<<< HEAD
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
=======
User provides 3 photos (top, bottom, shoes)
        ↓
Gemini Vision classifies each item
(category, color, pattern, style, fit, material)
        ↓
User inputs event type (casual, office, wedding...)
        ↓
Gemini scores outfit compatibility (0.0 → 1.0)
        ↓
Prints verdict, what works, what doesn't, styling tips
        ↓
Saves full results to outfit_result.json
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

---

## 📁 Project Structure

```
<<<<<<< HEAD
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
=======
digital-wardrobe-poc/
├── main.py                 # Main PoC script
├── requirements.txt        # Python dependencies
├── .env                    # API key (never committed)
├── .gitignore
├── README.md               # This file
└── sample_clothes/         # Put your test images here
    ├── top.jpg
    ├── bottom.jpg
    └── shoes.jpg
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
<<<<<<< HEAD
git clone https://github.com/your-username/digital-wardrobe.git
cd digital-wardrobe
=======
git clone https://github.com/your-username/digital-wardrobe-poc.git
cd digital-wardrobe-poc
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

### 2. Create a virtual environment
```bash
python -m venv venv
<<<<<<< HEAD
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
=======
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 4. Verify GPU (optional but recommended)
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
=======
### 4. Get your Gemini API Key
- Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Create a free API key (no credit card required)

### 5. Set up `.env`
```
GEMINI_API_KEY=your_gemini_api_key_here
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

---

## ▶️ Run

```bash
python main.py
```

<<<<<<< HEAD
When prompted:
```
=======
You'll be prompted to enter image paths and event type:
```
👗 Digital Wardrobe — Outfit Recommendation PoC
--------------------------------------------------
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
Path to TOP image    : sample_clothes/top.jpg
Path to BOTTOM image : sample_clothes/bottom.jpg
Path to SHOES image  : sample_clothes/shoes.jpg

<<<<<<< HEAD
Available events: casual, office, wedding, party, date night, gym
=======
Available events: casual, office, wedding, party, date night, gym, beach, formal
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
Event type: casual
```

---

## 📊 Sample Output

```
<<<<<<< HEAD
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
=======
👕 TOP CLASSIFICATION
  Type     : White Oxford Shirt
  Color    : White
  Pattern  : Solid
  Style    : Business Casual
  Events   : office, casual, date night

👕 BOTTOM CLASSIFICATION
  Type     : Navy Chinos
  Color    : Navy Blue
  Pattern  : Solid
  Style    : Smart Casual
  Events   : office, casual, date night, party

👟 SHOES CLASSIFICATION
  Type     : White Sneakers
  Color    : White
  Style    : Casual / Streetwear
  Events   : casual, gym, beach

============================================================
  👔  OUTFIT RECOMMENDATION FOR: CASUAL
============================================================
  Compatibility : [████████░░] 80%
  Verdict       : Great Match
  Event Fit     : Perfect for a casual outing

  ✅ What Works:
    • White shirt and navy chinos is a classic combination
    • White sneakers tie in with the shirt color

  ❌ What Doesn't Work:
    • None!

  💡 Styling Tips:
    • Roll up the shirt sleeves for a relaxed look
    • Add a minimalist watch to elevate the outfit
    • Tuck in the shirt for a cleaner silhouette
============================================================
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
```

---

<<<<<<< HEAD
=======
## ⚠️ Limitations (PoC Stage)

- Classification quality depends on image clarity — use well-lit photos
- Works best with one item clearly visible per photo
- Gemini free tier limited to 250 requests/day
- No persistent wardrobe storage yet — each run is independent

---

## 🗺️ Roadmap

- [x] Clothing classification via Gemini Vision
- [x] Outfit compatibility scoring
- [x] Event-based recommendation
- [ ] Replace Gemini classifier with custom trained ResNet model
- [ ] Persistent wardrobe storage (SQLite / Firebase)
- [ ] Streamlit UI with drag-and-drop upload
- [ ] Multi-outfit generation (top 3 combinations)
- [ ] Outfit history and favourites

---

>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde
## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
<<<<<<< HEAD
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
=======
| [Gemini 2.5 Flash](https://aistudio.google.com) | Vision classification + compatibility scoring |
| [Pillow](https://pillow.readthedocs.io) | Image loading and validation |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | API key management |
>>>>>>> f3660f7051c7e8315bcc14230b608393d0013fde

---

## 📄 License

MIT