"""
data/preprocess.py
──────────────────
Converts raw MovieLens 1M .dat files into the chronologically-ordered
user interaction sequences that BERT4Rec consumes.

Pipeline:
    1.  Load ratings.dat  →  (user_id, item_id, rating, timestamp)
    2.  Filter low-rating interactions (keep rating >= min_rating)
    3.  Encode users and items into 1-based contiguous integer IDs
        (0 is reserved for the [PAD] token; vocab_size = num_items + 1)
    4.  Sort each user's interactions by timestamp → chronological sequence
    5.  Drop users whose sequence is shorter than min_seq_len
    6.  Leave-one-out split:
            train  = seq[:-2]   (all but last two items)
            val    = seq[:-1]   (all but last item)  → predict seq[-2]
            test   = seq        (full sequence)      → predict seq[-1]
    7.  Save processed data as joblib artefacts + a stats JSON

Output files (inside --data_dir/processed/):
    train_seqs.pkl      dict[user_id → list[item_id]]
    val_seqs.pkl        dict[user_id → list[item_id]]
    test_seqs.pkl       dict[user_id → list[item_id]]
    item_encoder.pkl    sklearn LabelEncoder (original → 1-based int)
    user_encoder.pkl    sklearn LabelEncoder
    stats.json          dataset statistics
"""
from __future__ import annotations

import re
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PAD_TOKEN = 0          # reserved; never a real item id
MASK_TOKEN_OFFSET = 1  # item ids are 1-based; mask token = num_items + 1


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_ratings(raw_dir: Path, min_rating: float = 4.0) -> pd.DataFrame:
    """
    Parse ratings.dat (::  separator, no header) and keep interactions
    where rating >= *min_rating*.  Returns a DataFrame with columns:
    [user_id, item_id, rating, timestamp].
    """
    logger.info("Loading ratings from %s", raw_dir / "ratings.dat")
    df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    logger.info("  total interactions: %d", len(df))

    if min_rating > 0:
        df = df[df["rating"] >= min_rating].reset_index(drop=True)
        logger.info("  after rating filter (>= %.1f): %d", min_rating, len(df))

    return df

def build_item_features(
    raw_dir: Path,
    item_enc: LabelEncoder,
) -> tuple[dict[int, dict], dict]:
    """
    Build side features for each item from movies.dat.

    Returns:
        item_features: dict mapping encoded_item_id (1-based) → {
            "genres":  list[int]  multi-hot vector
            "decade":  int        decade bucket id
        }
        feature_stats: dict with num_genres, num_decades, etc.
    """
    logger.info("Building item side features …")

    # ── 1. Load raw movies metadata ──
    movies = load_movies(raw_dir)

    # ── 2. Build genre vocabulary ──
    all_genres = sorted(set(
        g for genre_str in movies["genres"]
        for g in genre_str.split("|")
    ))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)
    logger.info("  genres: %d → %s", num_genres, all_genres)

    # ── 3. Parse year from title and bucket into decades ──
    def extract_year(title: str) -> int:
        m = re.search(r"\((\d{4})\)", title)
        return int(m.group(1)) if m else 1990   # fallback

    movies["year"]   = movies["title"].apply(extract_year)
    movies["decade"] = (movies["year"] - 1900) // 10   # e.g. 1995 → 9

    all_decades = sorted(movies["decade"].unique())
    decade_to_idx = {d: i for i, d in enumerate(all_decades)}
    num_decades = len(all_decades)
    logger.info("  decades: %d → %s", num_decades, all_decades)

    # ── 4. Build mapping: raw_item_id → encoded_item_id (1-based) ──
    # item_enc was fitted on raw item ids; encoded = item_enc.transform([raw]) + 1
    valid_raw_ids = set(item_enc.classes_)

    # ── 5. Construct features for every item that survived filtering ──
    item_features: dict[int, dict] = {}

    for _, row in movies.iterrows():
        raw_id = row["item_id"]
        if raw_id not in valid_raw_ids:
            continue   # this movie had no high-rating interactions

        encoded_id = int(item_enc.transform([raw_id])[0]) + 1   # 1-based

        # multi-hot genre vector
        genres_multihot = [0] * num_genres
        for g in row["genres"].split("|"):
            genres_multihot[genre_to_idx[g]] = 1

        item_features[encoded_id] = {
            "genres": genres_multihot,
            "decade": decade_to_idx[row["decade"]],
        }

    logger.info("  built features for %d items", len(item_features))

    # ── 6. Stats ──
    feature_stats = {
        "num_genres":    num_genres,
        "num_decades":   num_decades,
        "genre_vocab":   all_genres,
        "decade_vocab":  [int(d) for d in all_decades],
    }

    return item_features, feature_stats

def load_movies(raw_dir: Path) -> pd.DataFrame:
    """Parse movies.dat.  Returns DataFrame with [item_id, title, genres]."""
    logger.info("Loading movie metadata from %s", raw_dir / "movies.dat")
    df = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        header=None,
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    return df


# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Fit LabelEncoders for user_id and item_id.
    Items are shifted by +1 so that 0 remains the PAD token.
    Returns (encoded_df, user_encoder, item_encoder).
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df = df.copy()
    df["user_id"] = user_enc.fit_transform(df["user_id"])
    df["item_id"] = item_enc.fit_transform(df["item_id"]) + 1  # 1-based

    logger.info("  unique users: %d", len(user_enc.classes_))
    logger.info("  unique items: %d  (vocab_size = %d incl. PAD)", len(item_enc.classes_), len(item_enc.classes_) + 1)
    return df, user_enc, item_enc


# ── Sequence building ──────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    min_seq_len: int = 5,
) -> dict[int, list[int]]:
    """
    Sort each user's ratings by timestamp and return a dict
    { user_id → [item_id, item_id, ...] }.
    Users with fewer than *min_seq_len* interactions are dropped.
    """
    logger.info("Building chronological interaction sequences …")
    df_sorted = df.sort_values(["user_id", "timestamp"])

    sequences: dict[int, list[int]] = {}
    for user_id, group in tqdm(df_sorted.groupby("user_id"), desc="users"):
        seq = group["item_id"].tolist()
        if len(seq) >= min_seq_len:
            sequences[user_id] = seq

    logger.info(
        "  kept %d users  (dropped %d with seq_len < %d)",
        len(sequences),
        df["user_id"].nunique() - len(sequences),
        min_seq_len,
    )
    return sequences


# ── Leave-one-out split ────────────────────────────────────────────────────────

def leave_one_out_split(
    sequences: dict[int, list[int]],
) -> tuple[dict, dict, dict]:
    """
    Standard leave-one-out split for sequential recommendation:
      train  seq[:-2]   model trains on all but the last two items
      val    seq[:-1]   model predicts seq[-2] (second-to-last)
      test   seq        model predicts seq[-1] (last item)

    Returns (train_seqs, val_seqs, test_seqs).
    """
    train_seqs: dict[int, list[int]] = {}
    val_seqs:   dict[int, list[int]] = {}
    test_seqs:  dict[int, list[int]] = {}

    for user_id, seq in sequences.items():
        train_seqs[user_id] = seq[:-2]
        val_seqs[user_id]   = seq[:-1]
        test_seqs[user_id]  = seq

    return train_seqs, val_seqs, test_seqs


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(
    sequences: dict[int, list[int]],
    item_enc: LabelEncoder,
) -> dict:
    """Return a dict of useful dataset statistics."""
    lengths = [len(s) for s in sequences.values()]
    return {
        "num_users":      len(sequences),
        "num_items":      len(item_enc.classes_),
        "vocab_size":     len(item_enc.classes_) + 1,   # +1 for PAD
        "total_interactions": sum(lengths),
        "seq_len_min":    int(np.min(lengths)),
        "seq_len_max":    int(np.max(lengths)),
        "seq_len_mean":   float(np.mean(lengths)),
        "seq_len_median": float(np.median(lengths)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess(
    data_dir:    str | Path = "data",
    min_rating:  float = 4.0,
    min_seq_len: int   = 5,
) -> Path:
    """
    Full preprocessing pipeline.  Reads raw/ and writes processed/.
    Returns the path to the processed/ directory.
    """
    data_dir    = Path(data_dir)
    raw_dir     = data_dir / "raw"
    proc_dir    = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & filter ──
    df = load_ratings(raw_dir, min_rating=min_rating)

    # ── 2. Encode ──
    df, user_enc, item_enc = encode_ids(df)

    # ── 3. Sequences ──
    sequences = build_sequences(df, min_seq_len=min_seq_len)

    # ── 4. Split ──
    train_seqs, val_seqs, test_seqs = leave_one_out_split(sequences)

    # ── 5. Stats ──
    stats = compute_stats(sequences, item_enc)

    # ── 5.5 Side features ──
    item_features, feature_stats = build_item_features(raw_dir, item_enc)
    stats.update(feature_stats)
    logger.info("Dataset stats: %s", json.dumps(
        {k: v for k, v in stats.items() if not isinstance(v, list)}, indent=2
    ))

    # ── 6. Save ──
    joblib.dump(item_features, proc_dir / "item_features.pkl")
    joblib.dump(train_seqs, proc_dir / "train_seqs.pkl")
    joblib.dump(val_seqs,   proc_dir / "val_seqs.pkl")
    joblib.dump(test_seqs,  proc_dir / "test_seqs.pkl")
    joblib.dump(user_enc,   proc_dir / "user_encoder.pkl")
    joblib.dump(item_enc,   proc_dir / "item_encoder.pkl")

    with open(proc_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved processed data to %s", proc_dir)
    return proc_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess MovieLens 1M for BERT4Rec")
    p.add_argument("--data_dir",    default="data",  help="Root data directory")
    p.add_argument("--min_rating",  type=float, default=4.0, help="Minimum rating to keep (default: 4.0)")
    p.add_argument("--min_seq_len", type=int,   default=5,   help="Minimum sequence length (default: 5)")
    p.add_argument("--log_level",   default="INFO",  choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s | %(message)s")
    preprocess(
        data_dir=args.data_dir,
        min_rating=args.min_rating,
        min_seq_len=args.min_seq_len,
    )
