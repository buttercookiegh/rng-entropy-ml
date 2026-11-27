"""Reproducible RNG ML study for the user's existing rng-entropy-ml uv project.

Run from the repository root:
    uv run python scripts/run_ml_analysis.py --task source
    uv run python scripts/run_ml_analysis.py --task next-bit

Only held-out test results are reported; PCA and scalers are fitted within training pipelines.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
# Predictor names mirror scripts/prepare_data.py. Exclude the derived
# Shannon-entropy statistic, source label, sequence ID and split assignment.
FEATURES = [
    "frequency_ones", "number_of_runs", "longest_run", "longest_zero_run",
    "longest_one_run", "transition_rate", "lag1_autocorrelation",
    "lag2_autocorrelation", "pair_00", "pair_01", "pair_10",
    "pair_11", "block_frequency_std",
]
BIT_FEATURES = [f"bit_t_minus_{i:02d}" for i in range(16, 0, -1)]


def ensure_output():
    FIG.mkdir(parents=True, exist_ok=True)


def record_scores(name, y_test, preds, probs, classes, elapsed):
    output = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "fit_predict_seconds": round(elapsed, 3),
    }
    if probs is not None:
        output["log_loss_bits"] = float(log_loss(y_test, probs, labels=classes) / np.log(2))
    return output


def save_barplot(df, title, filename, ykey="f1_macro"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(df["model"], df[ykey])
    ax.set_xlabel(ykey.replace("_", " "))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=160)
    plt.close(fig)


def source_task():
    path = ROOT / "data/processed/sequence_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; first run: uv run python scripts/prepare_data.py")
    df = pd.read_csv(path)
    x = df[FEATURES]
    y = df["source_label"].astype(int)
    split = df["split"].str.lower()
    train, test = split.eq("train"), split.eq("test")
    if not (train.any() and test.any() and not (train & test).any()):
        raise ValueError("Invalid train/test split field")
    x_train, x_test = x.loc[train], x.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar([str(n) for n in counts.index], counts.to_numpy())
    ax.set(xlabel="Source label", ylabel="Number of sequences", title="Generator-group frequencies")
    fig.tight_layout()
    fig.savefig(FIG / "source_class_counts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for label, group in df.groupby("source_label"):
        ax.hist(group["frequency_ones"], bins=26, alpha=0.45, label=f"Source {label}")
    ax.set(xlabel="Proportion of 1s in 100-bit sequence", ylabel="Count", title="Bit frequency by source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "source_frequency_histogram.png", dpi=160)
    plt.close(fig)

    # Exploratory PCA is fitted ONLY on training sequences. The labels are
    # used solely for plotting and are not included in PCA inputs.
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    pca = PCA().fit(train_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    ax.axhline(0.95, linestyle="--", linewidth=1)
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
           title="PCA explained variance (training data)")
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pca_explained_variance.png", dpi=160)
    plt.close(fig)

    pc2 = pca.transform(train_scaled)[:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label in sorted(y_train.unique()):
        take = y_train.to_numpy() == label
        ax.scatter(pc2[take, 0], pc2[take, 1], s=8, alpha=0.5, label=f"Source {label}")
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA projection (training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "pca_scatter.png", dpi=160)
    plt.close(fig)

    models = {
        "Class-prior baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                          ("model", LogisticRegression(max_iter=1000))]),
        "PCA + Logistic Regression": Pipeline([("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95)), ("model", LogisticRegression(max_iter=1000))]),
        "SVM (RBF kernel)": Pipeline([("scale", StandardScaler()),
                                     ("model", SVC(probability=True, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                                random_state=42, n_jobs=-1),
    }
    records = []
    best_score, best_name, best_preds = -1, None, None
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
        elapsed = time.perf_counter() - start
        row = record_scores(name, y_test, pred, prob, [1, 2, 3], elapsed)
        records.append(row)
        if row["f1_macro"] > best_score:
            best_name, best_score, best_preds = name, row["f1_macro"], pred
    scores = pd.DataFrame(records)
    scores.to_csv(OUT / "source_model_comparison.csv", index=False)
    save_barplot(scores, "Held-out source classification: macro-F1", "source_model_f1.png")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, best_preds, labels=[1,2,3]),
                           display_labels=[1,2,3]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(FIG / "source_confusion_matrix.png", dpi=160)
    plt.close(fig)
    (OUT / "source_notes.json").write_text(json.dumps({
        "sequence_count": int(len(df)), "train": int(train.sum()), "test": int(test.sum()),
        "predictor_count": len(FEATURES), "pca_95pct_components": n95,
        "best_test_macro_f1_model": best_name,
        "warning": "Source-group accuracy is not an entropy estimate. Test scores are exploratory; avoid repeatedly tuning on the test set."
    }, indent=2) + "\n")
    print("\nSOURCE CLASSIFICATION RESULTS\n", scores.to_string(index=False))
    print("PCA components explaining >=95% training variance:", n95)


def next_bit_task():
    base = ROOT / "data/processed"
    a, b = base / "next_bit_train.csv", base / "next_bit_test.csv"
    if not (a.exists() and b.exists()):
        raise FileNotFoundError("Missing next-bit CSVs; first run prepare_data.py")
    train, test = pd.read_csv(a), pd.read_csv(b)
    # Ensure no original sequence occurs in both datasets.
    overlap = set(train.sequence_id).intersection(test.sequence_id)
    if overlap:
        raise ValueError(f"Train/test leakage: {len(overlap)} shared sequence IDs")
    x_train = train[BIT_FEATURES]
    x_test = test[BIT_FEATURES]
    y_train = train["next_bit"].astype(int)
    y_test = test["next_bit"].astype(int)
    models = {
        "Prior-only baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "PCA (95%) + Logistic Regression": Pipeline([("pca", PCA(n_components=0.95)),
                                                      ("model", LogisticRegression(max_iter=500))]),
    }
    records = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)
        records.append(record_scores(name, y_test, pred, prob, [0, 1], time.perf_counter() - start))
    scores = pd.DataFrame(records)
    scores.to_csv(OUT / "next_bit_model_comparison.csv", index=False)
    save_barplot(scores, "Held-out next-bit prediction: accuracy", "next_bit_accuracy.png", "accuracy")
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.bar(scores["model"], scores["log_loss_bits"])
    ax.set(ylabel="Log loss (bits/prediction)", title="Predictive cross-entropy on held-out bits")
    ax.tick_params(axis="x", labelrotation=22)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "next_bit_logloss.png", dpi=160)
    plt.close(fig)
    p = float(y_test.mean())
    h = -sum(q * np.log2(q) for q in [p, 1-p] if q > 0)
    (OUT / "next_bit_notes.json").write_text(json.dumps({
        "test_bit_one_frequency": p, "test_marginal_shannon_entropy_bits": h,
        "original_train_sequences": int(train.sequence_id.nunique()),
        "original_test_sequences": int(test.sequence_id.nunique()),
        "warning": "Log loss is model cross-entropy, not a certified entropy estimate; 50% accuracy does not prove randomness."
    }, indent=2) + "\n")
    print("\nNEXT-BIT PREDICTION RESULTS\n", scores.to_string(index=False))
    print(f"Test marginal Shannon entropy: {h:.5f} bits/bit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["source", "next-bit", "all"], default="source")
    args = parser.parse_args()
    ensure_output()
    if args.task in ("source", "all"):
        source_task()
    if args.task in ("next-bit", "all"):
        next_bit_task()
    print("\nOutputs saved under", OUT)

if __name__ == "__main__":
    main()
