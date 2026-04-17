"""
Smart Wardrobe — Streamlit UI

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import io
from PIL import Image

from src.embeddings import load_feature_extractor, extract_embedding, DEVICE
from src.database import init_db, add_item, get_items, delete_item, count_items
from src.compatibility import score_outfit
from src.event_classifier import (
    load_event_classifier, compute_outfit_event_score, EVENT_TO_USAGE, MODEL_PATH
)
from src.recommender import get_top_outfits

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Smart Wardrobe", page_icon="👗", layout="wide")

# ── CACHED RESOURCES ──────────────────────────────────────────
@st.cache_resource
def get_extractor():
    return load_feature_extractor()

@st.cache_resource
def get_event_model():
    if Path(MODEL_PATH).exists():
        return load_event_classifier(MODEL_PATH, device=DEVICE)
    return None

def get_db():
    """Fresh connection each call — SQLite is lightweight and avoids threading issues."""
    return init_db("wardrobe.db")

# ── HELPERS ───────────────────────────────────────────────────
def show_thumbnail(thumbnail_bytes, width=100):
    """Display a thumbnail from bytes."""
    if thumbnail_bytes:
        img = Image.open(io.BytesIO(thumbnail_bytes))
        st.image(img, width=width)
    else:
        st.write("No image")

def score_bar(score, label="Score"):
    """Display a colored score bar."""
    if score >= 0.75:
        color = "🟢"
    elif score >= 0.55:
        color = "🟡"
    else:
        color = "🔴"
    st.markdown(f"**{label}:** {score:.0%} {color}")
    st.progress(min(score, 1.0))

# ── SIDEBAR NAVIGATION ───────────────────────────────────────
st.sidebar.title("👗 Smart Wardrobe")
page = st.sidebar.radio("Navigate", ["Upload", "My Wardrobe", "Recommend", "About"])

conn = get_db()
counts = count_items(conn)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Wardrobe:** {counts['top']} tops | {counts['bottom']} bottoms | {counts['shoes']} shoes")

# ── PAGE: UPLOAD ──────────────────────────────────────────────
if page == "Upload":
    st.title("Upload Clothing Item")
    st.write("Add a new item to your wardrobe by uploading a photo.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        name = st.text_input("Item name", placeholder="e.g. Blue Denim Jacket")
        category = st.selectbox("Category", ["top", "bottom", "shoes"])

    with col2:
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Preview", width=250)

    if st.button("Add to Wardrobe", type="primary", disabled=not uploaded_file or not name):
        with st.spinner("Extracting features..."):
            # Save temp file for embedding extraction
            temp_path = Path("temp_upload.jpg")
            img = Image.open(uploaded_file).convert("RGB")
            img.save(temp_path)

            extractor = get_extractor()
            embedding = extract_embedding(extractor, str(temp_path))
            item_id = add_item(conn, name, category, embedding, str(temp_path))

            temp_path.unlink()  # cleanup

        st.success(f"Added '{name}' to {category}s! (ID: {item_id})")
        st.cache_resource.clear()
        st.rerun()

# ── PAGE: MY WARDROBE ────────────────────────────────────────
elif page == "My Wardrobe":
    st.title("My Wardrobe")

    for cat, label in [("top", "Tops"), ("bottom", "Bottoms"), ("shoes", "Shoes")]:
        items = get_items(conn, cat)
        st.subheader(f"{label} ({len(items)})")

        if not items:
            st.info(f"No {label.lower()} yet. Upload some!")
            continue

        cols = st.columns(min(len(items), 5))
        for i, item in enumerate(items):
            with cols[i % 5]:
                show_thumbnail(item.get("thumbnail"), width=90)
                st.caption(item["name"])
                if st.button("Delete", key=f"del_{item['id']}"):
                    delete_item(conn, item["id"])
                    st.cache_resource.clear()
                    st.rerun()
        st.markdown("---")

# ── PAGE: RECOMMEND ───────────────────────────────────────────
elif page == "Recommend":
    st.title("Outfit Recommendations")

    counts = count_items(conn)
    if counts["top"] == 0 or counts["bottom"] == 0 or counts["shoes"] == 0:
        st.warning("You need at least 1 top, 1 bottom, and 1 pair of shoes to get recommendations. Upload some items first!")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            event = st.selectbox("Event type", list(EVENT_TO_USAGE.keys()))
        with col2:
            top_n = st.slider("Number of outfits", 1, 10, 3)

        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Scoring outfit combinations..."):
                event_model = get_event_model()
                results = get_top_outfits(
                    conn,
                    compat_weight=0.5,
                    event_weight=0.5 if event_model else 0.0,
                    event=event,
                    event_model=event_model,
                    device=DEVICE,
                    top_n=top_n,
                )

            if not results:
                st.error("No outfit combinations found.")
            else:
                if event_model is None:
                    st.info("Event classifier not trained yet. Ranking by compatibility only. Run `python -m training.train_event_model` to enable event-aware scoring.")

                for rank, outfit in enumerate(results, 1):
                    with st.container():
                        st.subheader(f"#{rank}")
                        cols = st.columns([1, 1, 1, 2])

                        with cols[0]:
                            st.markdown("**Top**")
                            show_thumbnail(outfit["top"].get("thumbnail"))
                            st.caption(outfit["top"]["name"])

                        with cols[1]:
                            st.markdown("**Bottom**")
                            show_thumbnail(outfit["bottom"].get("thumbnail"))
                            st.caption(outfit["bottom"]["name"])

                        with cols[2]:
                            st.markdown("**Shoes**")
                            show_thumbnail(outfit["shoes"].get("thumbnail"))
                            st.caption(outfit["shoes"]["name"])

                        with cols[3]:
                            st.markdown("**Scores**")
                            score_bar(outfit["scores"]["compatibility"], "Compatibility")
                            if event_model:
                                score_bar(outfit["scores"]["event_suitability"], f"Event ({event})")
                            score_bar(outfit["scores"]["final"], "Overall")

                    st.markdown("---")

# ── PAGE: ABOUT ───────────────────────────────────────────────
elif page == "About":
    st.title("About Smart Wardrobe")
    st.markdown("""
    **Smart Wardrobe** is a local ML-powered outfit recommender.

    ### How it works
    1. **Upload** clothing photos (tops, bottoms, shoes)
    2. **ResNet50** extracts a 2048-dimensional embedding from each image
    3. Embeddings are stored in a **SQLite** database
    4. When you request recommendations:
       - **Compatibility** is scored via cosine similarity between item embeddings
       - **Event suitability** is predicted by an MLP trained on 44k fashion items
       - Scores are combined to rank all outfit combinations
       - **Top N outfits** are returned

    ### Tech Stack
    - **PyTorch** — ResNet50 embeddings + event classifier
    - **SQLite** — wardrobe storage
    - **Streamlit** — this UI
    - **scikit-learn** — cosine similarity + evaluation metrics

    ### Event Types
    - Casual, Office, Wedding, Party, Date Night, Gym

    ---
    *Built with AI assistance from Claude (Anthropic).*
    """)
