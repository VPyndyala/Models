#!/usr/bin/env python3
"""
V2: Better baseline for rung score prediction.
- Uses BOTH:
  (A) Sentence embeddings per field (prompt/base/r1..r5) concatenated
  (B) TF-IDF char n-grams on the full concatenated text
- Optionally uses GroupShuffleSplit if a grouping column exists
- Trains one classifier per output
- Tunes threshold per output to maximize F1 on val split

Outputs:
- metrics.json
- predictions_test.csv
- model.joblib
"""

import os, sys, json, argparse
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

import joblib

TEXT_COLS = ["prompt", "base", "r1", "r2", "r3", "r4", "r5"]
TARGET_COLS = ["r1_score", "r2_score", "r3_score", "r4_score", "r5_score"]

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\x00", "").strip()

def assert_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")

def coerce_binary_targets(df: pd.DataFrame) -> np.ndarray:
    y = df[TARGET_COLS].copy()
    for c in TARGET_COLS:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    if y.isna().any().any():
        raise ValueError("Targets contain NaN/non-numeric values after coercion.")
    return (y.values > 0.5).astype(int)

def build_full_text(df: pd.DataFrame) -> List[str]:
    out = []
    for _, row in df.iterrows():
        chunks = []
        for c in TEXT_COLS:
            chunks.append(f"[{c.upper()}]\n{clean_text(row[c])}")
        out.append("\n\n".join(chunks))
    return out

def compute_label_stats(y: np.ndarray) -> Dict:
    stats = {}
    for i, col in enumerate(TARGET_COLS):
        p = float(y[:, i].mean())
        stats[col] = {"pos_rate": p, "neg_rate": 1.0 - p}
    return stats

def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    # Scan thresholds to maximize F1
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        yp = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

def metrics_for_head(y_true, y_pred, y_prob) -> Dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": None,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_plus_r2synth_graded_life_scored.csv")
    ap.add_argument("--out_dir", default="rung_score_model_out_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)  # fraction of train used for threshold tuning
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--group_col", default="", help="Optional grouping column (e.g., episode_id). If empty, auto-detect.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    assert_columns(df, TEXT_COLS, "text")
    assert_columns(df, TARGET_COLS, "targets")

    for c in TEXT_COLS:
        df[c] = df[c].apply(clean_text)

    y = coerce_binary_targets(df)
    label_stats = compute_label_stats(y)

    # Auto-detect a plausible grouping column if not provided
    group_col = args.group_col.strip()
    if not group_col:
        for cand in ["episode_id", "episode", "group_id", "story_id", "id"]:
            if cand in df.columns:
                group_col = cand
                break
    groups = df[group_col].values if group_col and group_col in df.columns else None

    # Build texts
    full_text = build_full_text(df)

    # Split: prefer group split if groups exist and are not all-unique
    idx = np.arange(len(df))
    if groups is not None and len(np.unique(groups)) < len(groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(gss.split(idx, groups=groups))
    else:
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)

    # Further split train into train/val for threshold tuning
    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=args.seed, shuffle=True)

    X_train_text = [full_text[i] for i in train_idx]
    X_val_text   = [full_text[i] for i in val_idx]
    X_test_text  = [full_text[i] for i in test_idx]

    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    # Sentence embeddings per FIELD, concatenated
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install -U sentence-transformers", file=sys.stderr)
        sys.exit(1)

    embedder = SentenceTransformer(args.embed_model)

    def embed_fields(indices: np.ndarray) -> np.ndarray:
        # Encode each field separately, then concatenate
        mats = []
        for c in TEXT_COLS:
            texts = [df.loc[i, c] for i in indices]
            emb = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            mats.append(emb)
        return np.concatenate(mats, axis=1)

    X_train_emb = embed_fields(train_idx)
    X_val_emb   = embed_fields(val_idx)
    X_test_emb  = embed_fields(test_idx)

    # TF-IDF char n-grams on full text (robust to templates / short cues)
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf   = tfidf.transform(X_val_text)
    X_test_tfidf  = tfidf.transform(X_test_text)

    # Combine: [TFIDF | EMB]
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_emb)])
    X_val   = hstack([X_val_tfidf,   csr_matrix(X_val_emb)])
    X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_emb)])

    # Train one classifier per head so we can threshold-tune independently
    clfs = []
    thresholds = []
    val_best_f1s = []
    per_output = {}

    for j, col in enumerate(TARGET_COLS):
        clf = LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
        )
        clf.fit(X_train, y_train[:, j])

        # probs on val
        val_prob = clf.predict_proba(X_val)[:, 1]
        t, best_f1 = pick_best_threshold(y_val[:, j], val_prob)
        thresholds.append(t)
        val_best_f1s.append(best_f1)

        # test
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= t).astype(int)

        per_output[col] = {
            "threshold": t,
            "val_best_f1_at_threshold": best_f1,
            "test": metrics_for_head(y_test[:, j], test_pred, test_prob),
        }

        clfs.append(clf)

    macro = {
        "accuracy": float(np.mean([per_output[c]["test"]["accuracy"] for c in TARGET_COLS])),
        "f1": float(np.mean([per_output[c]["test"]["f1"] for c in TARGET_COLS])),
        "precision": float(np.mean([per_output[c]["test"]["precision"] for c in TARGET_COLS])),
        "recall": float(np.mean([per_output[c]["test"]["recall"] for c in TARGET_COLS])),
        "auc": float(np.mean([per_output[c]["test"]["auc"] for c in TARGET_COLS if per_output[c]["test"]["auc"] is not None]))
               if any(per_output[c]["test"]["auc"] is not None for c in TARGET_COLS) else None,
    }

    metrics = {
        "group_col_used": group_col if groups is not None else None,
        "label_stats": label_stats,
        "macro_test": macro,
        "per_output": per_output,
    }

    print("\n=== Label Stats (pos rate) ===")
    for c in TARGET_COLS:
        print(f"{c}: {metrics['label_stats'][c]['pos_rate']:.3f}")

    print("\n=== Macro Test Metrics ===")
    for k, v in macro.items():
        print(f"{k}: {v}")

    print("\n=== Per-Output (test) ===")
    for c in TARGET_COLS:
        m = per_output[c]
        print(f"\n[{c}] threshold={m['threshold']:.2f} val_best_f1={m['val_best_f1_at_threshold']:.3f}")
        for k, v in m["test"].items():
            print(f"  {k}: {v}")

    # Save
    joblib.dump(
        {
            "embed_model": args.embed_model,
            "embedder": embedder,
            "tfidf": tfidf,
            "classifiers": clfs,
            "thresholds": thresholds,
            "text_cols": TEXT_COLS,
            "target_cols": TARGET_COLS,
            "group_col_used": group_col if groups is not None else None,
        },
        os.path.join(args.out_dir, "model.joblib"),
    )

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV on test set
    out = df.loc[test_idx, TEXT_COLS + TARGET_COLS].copy()
    for j, col in enumerate(TARGET_COLS):
        test_prob = clfs[j].predict_proba(X_test)[:, 1]
        out[col + "_prob"] = test_prob
        out[col + "_pred"] = (test_prob >= thresholds[j]).astype(int)
    out.to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)

    print(f"\nSaved to: {args.out_dir}/model.joblib, metrics.json, predictions_test.csv")


if __name__ == "__main__":
    main()
