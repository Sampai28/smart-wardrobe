"""
SQLite wardrobe store — stores clothing items with embeddings and metadata.
"""

import sqlite3
import numpy as np
import io
from pathlib import Path
from PIL import Image

DB_PATH = "wardrobe.db"


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL CHECK(category IN ('top', 'bottom', 'shoes')),
            gender      TEXT NOT NULL DEFAULT 'Unisex',
            embedding   BLOB NOT NULL,
            thumbnail   BLOB,
            added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
    """)
    # Migrate existing databases that lack the gender column
    try:
        conn.execute("ALTER TABLE items ADD COLUMN gender TEXT NOT NULL DEFAULT 'Unisex'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    return conn


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _make_thumbnail(image_path: str, size: tuple = (100, 100)) -> bytes:
    """Create a small JPEG thumbnail from an image file."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def add_item(conn: sqlite3.Connection, name: str, category: str,
             embedding: np.ndarray, image_path: str = None,
             gender: str = "Unisex") -> int:
    """Add a clothing item to the wardrobe. Returns the new item ID."""
    thumbnail = _make_thumbnail(image_path) if image_path else None
    cursor = conn.execute(
        "INSERT INTO items (name, category, gender, embedding, thumbnail) VALUES (?, ?, ?, ?, ?)",
        (name, category, gender, _serialize_embedding(embedding), thumbnail)
    )
    conn.commit()
    return cursor.lastrowid


def get_items(conn: sqlite3.Connection, category: str = None) -> list:
    """Get all items, optionally filtered by category."""
    if category:
        rows = conn.execute("SELECT * FROM items WHERE category = ? ORDER BY added_at DESC", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM items ORDER BY added_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_item(conn: sqlite3.Connection, item_id: int) -> dict:
    """Get a single item by ID."""
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row) if row else None


def delete_item(conn: sqlite3.Connection, item_id: int) -> bool:
    """Delete an item by ID. Returns True if deleted."""
    cursor = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    return cursor.rowcount > 0


def count_items(conn: sqlite3.Connection) -> dict:
    """Count items by category."""
    rows = conn.execute("SELECT category, COUNT(*) as cnt FROM items GROUP BY category").fetchall()
    counts = {r["category"]: r["cnt"] for r in rows}
    return {"top": counts.get("top", 0), "bottom": counts.get("bottom", 0), "shoes": counts.get("shoes", 0)}


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a Row to a dict with deserialized embedding."""
    d = dict(row)
    d["embedding"] = _deserialize_embedding(d["embedding"])
    if "gender" not in d:
        d["gender"] = "Unisex"
    return d
