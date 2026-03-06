# smart-wardrobe# 👗 Digital Wardrobe — Outfit Recommendation PoC

A Python proof-of-concept that classifies clothing items from photos and recommends outfit combinations based on the event type — using Google Gemini Vision.

---

## 🧠 How It Works

```
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
```

---

## 📁 Project Structure

```
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
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/digital-wardrobe-poc.git
cd digital-wardrobe-poc
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your Gemini API Key
- Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Create a free API key (no credit card required)

### 5. Set up `.env`
```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## ▶️ Run

```bash
python main.py
```

You'll be prompted to enter image paths and event type:
```
👗 Digital Wardrobe — Outfit Recommendation PoC
--------------------------------------------------
Path to TOP image    : sample_clothes/top.jpg
Path to BOTTOM image : sample_clothes/bottom.jpg
Path to SHOES image  : sample_clothes/shoes.jpg

Available events: casual, office, wedding, party, date night, gym, beach, formal
Event type: casual
```

---

## 📊 Sample Output

```
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
```

---

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

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Gemini 2.5 Flash](https://aistudio.google.com) | Vision classification + compatibility scoring |
| [Pillow](https://pillow.readthedocs.io) | Image loading and validation |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | API key management |

---

## 📄 License

MIT