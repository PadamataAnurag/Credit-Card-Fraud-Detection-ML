"""
=============================================================================
  ML ASSIGNMENT 2 — TRAIN & EVALUATE ALL 5 MODELS
  Author   : PADAMATA ANURAG
  Roll No  : 2026ak05613
  Email    : 2026ak05613@wilp.bits-pilani.ac.in
  Course   : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Dataset  : Credit Card Fraud Detection 2026
  Platform : BITS Virtual Lab

  Description:
    Trains 5 ML classifiers from scratch on the Credit Card Fraud dataset
    and evaluates each using 6 metrics: Accuracy, AUC, Precision, Recall,
    F1 Score, and Matthews Correlation Coefficient (MCC).

    Models implemented from scratch (NO sklearn model classes used):
      1. Logistic Regression   (Gradient Descent)
      2. Decision Tree         (CART — Gini Impurity)
      3. K-Nearest Neighbor    (Euclidean Distance, batched)
      4. Gaussian Naive Bayes  (Gaussian Likelihood)
      5. SVM                   (Hinge Loss + Mini-Batch SGD)
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

# ── Project path setup ──────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, "..", "Dataset",
                            "credit_card_fraud_detection_2026.csv")
SAVED_DIR    = os.path.join(PROJECT_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)

from models import (LogisticRegressionScratch, DecisionTreeScratch,
                    KNNScratch, GaussianNaiveBayesScratch, SVMScratch)
from utils  import MetricsScratch, Preprocessor


# ============================================================================
# CONSTANTS
# ============================================================================

RANDOM_SEED  = 42
TEST_SIZE    = 0.20        # 80/20 train-test split
KNN_SUBSET   = 10_000      # KNN training subset (speed optimisation for 100k rows)

np.random.seed(RANDOM_SEED)

BANNER = "=" * 70


# ============================================================================
# HELPERS
# ============================================================================

def print_banner(title):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def train_test_split_manual(X, y, test_size=0.20, seed=42):
    """
    Manual train/test split — no sklearn.
    Stratified to preserve class ratio.
    """
    rng      = np.random.default_rng(seed)
    classes  = np.unique(y)
    train_idx, test_idx = [], []

    for c in classes:
        c_idx      = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test     = max(1, int(len(c_idx) * test_size))
        test_idx.extend(c_idx[:n_test].tolist())
        train_idx.extend(c_idx[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])


def evaluate_model(name, model, X_test, y_test):
    """Run prediction + compute all 6 metrics."""
    y_pred   = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]   # P(class=1)
    metrics  = MetricsScratch.compute_all(y_test, y_pred, y_scores, positive_class=1)

    print(f"\n  {'Metric':<12} {'Value':>10}")
    print(f"  {'-'*25}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>10.6f}")

    # Confusion matrix
    cm = MetricsScratch.confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    return metrics


def save_model(model, name):
    """Save trained model to saved_models/ using joblib."""
    path = os.path.join(SAVED_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"  [SAVED] -> {path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # ── 1. LOAD DATA ────────────────────────────────────────────────────────
    print_banner("STEP 1 — Loading Dataset")
    print(f"  Dataset : {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Shape   : {df.shape}")
    print(f"  Classes : {df['is_fraud'].value_counts().to_dict()}")

    # ── 2. PREPROCESS ───────────────────────────────────────────────────────
    print_banner("STEP 2 — Preprocessing")
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    print(f"  Features after preprocessing : {X.shape[1]}")
    print(f"  Total samples                : {X.shape[0]}")
    print(f"  Class 0 (Legitimate)         : {(y==0).sum()}")
    print(f"  Class 1 (Fraud)              : {(y==1).sum()}")

    # ── 3. TRAIN / TEST SPLIT ───────────────────────────────────────────────
    print_banner("STEP 3 — Stratified Train/Test Split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=TEST_SIZE, seed=RANDOM_SEED)
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    # Save test data for Streamlit app
    test_df           = pd.DataFrame(X_test,
                                     columns=preprocessor.feature_cols_)
    test_df["is_fraud"] = y_test
    test_csv_path     = os.path.join(PROJECT_DIR, "..", "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Test data saved -> {test_csv_path}")

    # Save preprocessor (needed by Streamlit app)
    save_model(preprocessor, "preprocessor")

    # ── 4. DEFINE MODELS ────────────────────────────────────────────────────
    models_config = {
        "logistic_regression": {
            "model"  : LogisticRegressionScratch(
                            learning_rate=0.1,
                            n_iterations=500,
                            threshold=0.3,          # lower threshold for imbalanced data
                            class_weight='balanced'),# up-weight minority (fraud) class
            "verbose": True,
        },
        "decision_tree": {
            "model"  : DecisionTreeScratch(
                            max_depth=12,
                            min_samples_split=10,
                            n_thresholds=20),
            "verbose": True,
        },
        "knn": {
            "model"  : KNNScratch(k=7, batch_size=500),
            "verbose": False,
            "use_subset": True,     # use KNN_SUBSET samples for speed
        },
        "naive_bayes": {
            "model"  : GaussianNaiveBayesScratch(var_smoothing=1e-9),
            "verbose": True,
        },
        "svm": {
            "model"  : SVMScratch(
                            C=1.0,
                            learning_rate=0.001,
                            n_epochs=30,
                            batch_size=512,
                            class_weight='balanced'),# up-weight minority (fraud) class
            "verbose": True,
        },
    }

    # ── 5. TRAIN & EVALUATE EACH MODEL ──────────────────────────────────────
    results = {}

    for model_name, cfg in models_config.items():
        model   = cfg["model"]
        verbose = cfg.get("verbose", False)

        print_banner(f"MODEL: {model_name.upper().replace('_', ' ')}")

        # Some models (KNN) use a training subset for speed
        if cfg.get("use_subset"):
            subset_idx = np.random.choice(
                len(X_train), min(KNN_SUBSET, len(X_train)), replace=False)
            X_tr = X_train[subset_idx]
            y_tr = y_train[subset_idx]
            print(f"  [KNN] Using {len(X_tr):,} training samples "
                  f"(subset of {len(X_train):,})")
        else:
            X_tr, y_tr = X_train, y_train

        # ── Train ──
        print(f"\n  Training {model_name} ...")
        t0 = time.time()
        model.fit(X_tr, y_tr, verbose=verbose) if verbose else model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        print(f"  Training time : {train_time:.2f}s")

        # ── Evaluate ──
        print(f"\n  Evaluating on test set ({len(X_test):,} samples) ...")
        t0 = time.time()
        metrics    = evaluate_model(model_name, model, X_test, y_test)
        eval_time  = time.time() - t0
        print(f"  Evaluation time : {eval_time:.2f}s")

        results[model_name] = metrics

        # ── Save model ──
        save_model(model, model_name)

    # ── 6. COMPARISON TABLE ─────────────────────────────────────────────────
    print_banner("RESULTS — MODEL COMPARISON TABLE")
    print(f"\n  {'Model':<30} {'Accuracy':>10} {'AUC':>10} {'Precision':>10}"
          f" {'Recall':>10} {'F1':>10} {'MCC':>10}")
    print(f"  {'-'*90}")

    model_display = {
        "logistic_regression": "Logistic Regression",
        "decision_tree"      : "Decision Tree",
        "knn"                : "KNN (k=7)",
        "naive_bayes"        : "Naive Bayes (Gaussian)",
        "svm"                : "SVM (Linear SGD)",
    }

    best_model = max(results, key=lambda m: results[m]["F1"])

    for k, metrics in results.items():
        name = model_display.get(k, k)
        star = " *" if k == best_model else ""
        print(f"  {name+star:<30} "
              f"{metrics['Accuracy']:>10.4f} "
              f"{metrics['AUC']:>10.4f} "
              f"{metrics['Precision']:>10.4f} "
              f"{metrics['Recall']:>10.4f} "
              f"{metrics['F1']:>10.4f} "
              f"{metrics['MCC']:>10.4f}")

    print(f"\n  * = Best model by F1 Score")

    # ── 7. SAVE RESULTS CSV ─────────────────────────────────────────────────
    results_path = os.path.join(PROJECT_DIR, "results", "metrics_summary.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    rows = []
    for k, metrics in results.items():
        row = {"Model": model_display.get(k, k)}
        row.update(metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n  Results saved -> {results_path}")

    print_banner("TRAINING COMPLETE!")
    print(f"""
  Student   : PADAMATA ANURAG
  Roll No   : 2026ak05613
  Email     : 2026ak05613@wilp.bits-pilani.ac.in
  Course    : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Dataset   : Credit Card Fraud Detection 2026
  Models    : 5 (all implemented from scratch)
  Platform  : BITS Virtual Lab
""")


# ============================================================================
if __name__ == "__main__":
    main()
