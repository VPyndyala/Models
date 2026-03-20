import math
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# Config
# ============================================================

CSV_PATH = "reasoning_7d_balanced_augmented.csv"
MODEL_OUT_PATH = "reasoning_model_bundle.joblib"
SEED = 42
N_SPLITS = 5

LABEL_COLS = [
    "relevance_to_prompt",
    "directly_addresses_question",
    "step_by_step_or_structured_reasoning",
    "uses_justification_or_explanation",
    "internally_consistent",
    "acknowledges_uncertainty_or_limits_when_needed",
    "sufficiently_complete_for_prompt",
]

TEXT_MAX_FEATURES = 12000
TEXT_NGRAM_RANGE = (1, 2)

# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def safe_int01(x: Any) -> int:
    try:
        v = int(float(x))
        return 1 if v == 1 else 0
    except Exception:
        return 0

def build_text(user_prompt: str, assistant_response: str) -> str:
    return (
        f"[PROMPT]\n{str(user_prompt).strip()}\n\n"
        f"[RESPONSE]\n{str(assistant_response).strip()}"
    )

def extract_numeric_features(prompt: str, response: str):
    prompt = prompt or ""
    response = response or ""

    prompt_words = prompt.split()
    response_words = response.split()

    def count_substrings(s: str, subs):
        s_low = s.lower()
        return sum(s_low.count(x) for x in subs)

    step_markers = count_substrings(
        response,
        ["first", "second", "third", "therefore", "thus", "because", "if", "then", "finally", "step"]
    )
    hedge_markers = count_substrings(
        response,
        ["maybe", "perhaps", "possibly", "might", "could", "likely", "probably"]
    )
    contradiction_markers = count_substrings(
        response,
        ["however", "but", "although", "yet", "nevertheless"]
    )

    return [
        len(prompt),
        len(response),
        len(prompt_words),
        len(response_words),
        response.count("\n"),
        response.count("?"),
        response.count(":"),
        response.count(";"),
        sum(ch.isdigit() for ch in response),
        sum(ch in "=+-/*^" for ch in response),
        step_markers,
        hedge_markers,
        contradiction_markers,
        1.0 if "final answer" in response.lower() else 0.0,
        1.0 if any(x in response for x in ["1.", "2.", "3.", "- ", "* "]) else 0.0,
    ]

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

# ============================================================
# Load
# ============================================================

set_seed(SEED)

df = pd.read_csv(CSV_PATH)

required_cols = ["user_prompt", "assistant_response", "reasoning_score_7d"] + LABEL_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for c in LABEL_COLS:
    df[c] = df[c].apply(safe_int01)

df["reasoning_score_7d"] = pd.to_numeric(df["reasoning_score_7d"], errors="coerce")
df = df.dropna(subset=["user_prompt", "assistant_response", "reasoning_score_7d"]).reset_index(drop=True)

df["text"] = df.apply(
    lambda r: build_text(r["user_prompt"], r["assistant_response"]),
    axis=1,
)

X_num = np.array(
    [extract_numeric_features(p, r) for p, r in zip(df["user_prompt"], df["assistant_response"])],
    dtype=float,
)
Y = df[LABEL_COLS].values.astype(int)
y_score = df["reasoning_score_7d"].values.astype(float)

print(f"Rows: {len(df)}")
print(f"Score mean/std: {y_score.mean():.4f} / {y_score.std():.4f}")

# ============================================================
# 5-fold CV evaluation
# ============================================================

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

per_label_f1 = {c: [] for c in LABEL_COLS}
per_label_acc = {c: [] for c in LABEL_COLS}
micro_f1s = []
macro_f1s = []
exact_match_accs = []

score_metrics_probs = []
score_metrics_binary = []
score_metrics_ridge = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    score_train = y_score[train_idx]
    score_test = y_score[test_idx]

    X_num_train = X_num[train_idx]
    X_num_test = X_num[test_idx]

    vectorizer = TfidfVectorizer(
        max_features=TEXT_MAX_FEATURES,
        ngram_range=TEXT_NGRAM_RANGE,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    X_text_train = vectorizer.fit_transform(train_df["text"])
    X_text_test = vectorizer.transform(test_df["text"])

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train = hstack([X_text_train, csr_matrix(X_num_train_scaled)], format="csr")
    X_test = hstack([X_text_test, csr_matrix(X_num_test_scaled)], format="csr")

    clf = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
        )
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    Y_prob = clf.predict_proba(X_test)

    pred_score_from_labels = Y_pred.sum(axis=1).astype(float)
    pred_score_from_probs = Y_prob.sum(axis=1).astype(float)

    reg = Ridge(alpha=3.0, random_state=SEED)
    reg.fit(X_train, score_train)
    pred_score_direct = np.clip(reg.predict(X_test), 0.0, 7.0)

    for i, col in enumerate(LABEL_COLS):
        per_label_acc[col].append(accuracy_score(Y_test[:, i], Y_pred[:, i]))
        per_label_f1[col].append(f1_score(Y_test[:, i], Y_pred[:, i], zero_division=0))

    micro_f1s.append(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
    macro_f1s.append(f1_score(Y_test, Y_pred, average="macro", zero_division=0))
    exact_match_accs.append(accuracy_score(Y_test, Y_pred))

    score_metrics_binary.append(regression_metrics(score_test, pred_score_from_labels))
    score_metrics_probs.append(regression_metrics(score_test, pred_score_from_probs))
    score_metrics_ridge.append(regression_metrics(score_test, pred_score_direct))

    print(
        f"fold={fold} "
        f"micro_f1={micro_f1s[-1]:.4f} "
        f"macro_f1={macro_f1s[-1]:.4f} "
        f"exact={exact_match_accs[-1]:.4f} "
        f"r2_prob={score_metrics_probs[-1]['r2']:.4f} "
        f"r2_ridge={score_metrics_ridge[-1]['r2']:.4f}"
    )

# ============================================================
# Report
# ============================================================

print("\n===== 7-LABEL CLASSIFIER =====")
print(f"micro_f1: {np.mean(micro_f1s):.4f} ± {np.std(micro_f1s):.4f}")
print(f"macro_f1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
print(f"exact_match_acc: {np.mean(exact_match_accs):.4f} ± {np.std(exact_match_accs):.4f}")

print("\n===== PER-LABEL METRICS =====")
for col in LABEL_COLS:
    print(
        f"{col}: "
        f"acc={np.mean(per_label_acc[col]):.4f} ± {np.std(per_label_acc[col]):.4f}, "
        f"f1={np.mean(per_label_f1[col]):.4f} ± {np.std(per_label_f1[col]):.4f}"
    )

def summarize_reg(metrics_list, name):
    print(f"\n===== {name} =====")
    for k in ["mae", "rmse", "r2"]:
        vals = [m[k] for m in metrics_list]
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

summarize_reg(score_metrics_binary, "SCORE FROM SUM OF BINARY PREDS")
summarize_reg(score_metrics_probs, "SCORE FROM SUM OF PROBABILITIES")
summarize_reg(score_metrics_ridge, "DIRECT RIDGE SCORE")

# ============================================================
# Train final model on full dataset
# ============================================================

print("\n===== TRAINING FINAL MODEL ON FULL DATA =====")

vectorizer = TfidfVectorizer(
    max_features=TEXT_MAX_FEATURES,
    ngram_range=TEXT_NGRAM_RANGE,
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
)

X_text = vectorizer.fit_transform(df["text"])

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

X_all = hstack([X_text, csr_matrix(X_num_scaled)], format="csr")

clf = OneVsRestClassifier(
    LogisticRegression(
        C=2.0,
        max_iter=4000,
        class_weight="balanced",
        solver="liblinear",
    )
)
clf.fit(X_all, Y)

reg = Ridge(alpha=3.0, random_state=SEED)
reg.fit(X_all, y_score)

bundle = {
    "vectorizer": vectorizer,
    "scaler": scaler,
    "clf": clf,
    "reg": reg,
    "label_cols": LABEL_COLS,
    "text_max_features": TEXT_MAX_FEATURES,
    "text_ngram_range": TEXT_NGRAM_RANGE,
}

joblib.dump(bundle, MODEL_OUT_PATH)

print(f"Saved model bundle to: {MODEL_OUT_PATH}")
print("Bundle keys:", list(bundle.keys()))