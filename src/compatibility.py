"""
Compatibility scoring - cosine similarity between clothing embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def score_outfit(top_emb: np.ndarray, bottom_emb: np.ndarray, shoes_emb: np.ndarray) -> dict:
    """Compute pairwise cosine similarity compatibility score for a single outfit."""
    t = top_emb.reshape(1, -1)
    b = bottom_emb.reshape(1, -1)
    s = shoes_emb.reshape(1, -1)

    sim_tb = float(cosine_similarity(t, b)[0][0])
    sim_ts = float(cosine_similarity(t, s)[0][0])
    sim_bs = float(cosine_similarity(b, s)[0][0])
    avg = (sim_tb + sim_ts + sim_bs) / 3
    score = (avg + 1) / 2

    return {
        "top_bottom_sim": round(sim_tb, 4),
        "top_shoes_sim": round(sim_ts, 4),
        "bottom_shoes_sim": round(sim_bs, 4),
        "avg_similarity": round(avg, 4),
        "compatibility_score": round(score, 4),
    }


def score_outfits_batch(tops: np.ndarray, bottoms: np.ndarray, shoes: np.ndarray) -> np.ndarray:
    """Score multiple outfits at once. Returns (N,) array of scores in [0, 1]."""
    def row_cosine(a, b):
        dot = np.sum(a * b, axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        return dot / (norm_a * norm_b + 1e-8)

    sim_tb = row_cosine(tops, bottoms)
    sim_ts = row_cosine(tops, shoes)
    sim_bs = row_cosine(bottoms, shoes)
    avg = (sim_tb + sim_ts + sim_bs) / 3
    return (avg + 1) / 2
