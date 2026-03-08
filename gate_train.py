#!/usr/bin/env python3
"""
Train a GATE model (apply_grader: 0/1) from gate_dataset.csv.

Goal:
- apply_grader = 1 for decision-making / research / personal-info (high-stakes)
- apply_grader = 0 for low-stakes chatter / low-consequence questions

Input:  CSV with columns: id, kind, apply_grader, prompt
Model:  TF-IDF char ngrams + SentenceTransformer embedding -> LogisticRegression
Output:
- gate_model_out/model.joblib
- gate_model_out/metrics.json
- gate_model_out/predictions_test.csv

Usage:
  pip install -U pandas numpy scipy scikit-learn sentence-transformers joblib
  python train_gate.py --csv gate_dataset.csv
"""

import os
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

import joblib


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\x00", "").strip()
    return " ".join(s.split())


def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="gate_dataset.csv")
    ap.add_argument("--out_dir", default="gate_model_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)  # fraction of train used for threshold tuning
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_features", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    for col in ["prompt", "apply_grader"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["prompt"] = df["prompt"].apply(clean_text)
    df["apply_grader"] = pd.to_numeric(df["apply_grader"], errors="coerce")
    if df["apply_grader"].isna().any():
        raise ValueError("apply_grader has NaN/non-numeric values.")

    # Ensure 0/1
    y = (df["apply_grader"].values > 0.5).astype(int)
    X_text = df["prompt"].tolist()

    # Split train/test, then train/val inside train
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=args.seed, shuffle=True)

    X_train_text = [X_text[i] for i in train_idx]
    X_val_text   = [X_text[i] for i in val_idx]
    X_test_text  = [X_text[i] for i in test_idx]

    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    # TF-IDF char ngrams
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=args.max_features,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf   = tfidf.transform(X_val_text)
    X_test_tfidf  = tfidf.transform(X_test_text)

    # Embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Missing sentence-transformers. Install: pip install -U sentence-transformers")

    embedder = SentenceTransformer(args.embed_model)
    X_train_emb = embedder.encode(X_train_text, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    X_val_emb   = embedder.encode(X_val_text,   show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    X_test_emb  = embedder.encode(X_test_text,  show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_emb)])
    X_val   = hstack([X_val_tfidf,   csr_matrix(X_val_emb)])
    X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_emb)])

    # Classifier
    clf = LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Threshold tuning on val
    val_prob = clf.predict_proba(X_val)[:, 1]
    threshold, val_best_f1 = pick_best_threshold(y_val, val_prob)

    # Test metrics
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics: Dict = {
        "label_pos_rate": float(y.mean()),
        "threshold": float(threshold),
        "val_best_f1_at_threshold": float(val_best_f1),
        "test": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) == 2 else None,
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        }
    }

    print("\n=== Gate Model Metrics (test) ===")
    for k, v in metrics["test"].items():
        print(f"{k}: {v}")
    print(f"\nThreshold: {metrics['threshold']} (val best f1={metrics['val_best_f1_at_threshold']})")

    # Save artifacts
    joblib.dump(
        {
            "tfidf": tfidf,
            "embedder": embedder,
            "classifier": clf,
            "threshold": threshold,
        },
        os.path.join(args.out_dir, "model.joblib"),
    )

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    out = df.loc[test_idx, ["kind", "apply_grader", "prompt"]].copy() if "kind" in df.columns else df.loc[test_idx, ["apply_grader", "prompt"]].copy()
    out["apply_grader_prob"] = test_prob
    out["apply_grader_pred"] = test_pred
    out.to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)

    print(f"\nSaved to: {args.out_dir}/model.joblib, metrics.json, predictions_test.csv")


if __name__ == "__main__":
    main()
