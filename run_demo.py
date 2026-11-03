"""
=============================================================================
  ML ASSIGNMENT 2 — LIVE DEMO & EVALUATION SHOWCASE
  Student: PADAMATA ANURAG | Roll No: 2026ak05613
  BITS Pilani WILP — M.Tech (AIML)
=============================================================================
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models import (
    LogisticRegressionScratch,
    DecisionTreeScratch,
    KNNScratch,
    GaussianNaiveBayesScratch,
    SVMScratch,
)
from utils import MetricsScratch, Preprocessor

def main():
    print("=" * 95)
    print("  [BITS PILANI] - WORK INTEGRATED LEARNING PROGRAMMES (WILP)")
    print("  Course   : Machine Learning (M.Tech AIML) | Assignment - 2")
    print("  Student  : PADAMATA ANURAG | Roll No: 2026ak05613")
    print("  Email    : 2026ak05613@wilp.bits-pilani.ac.in")
    print("  Dataset  : Credit Card Fraud Detection (100,000 instances, 34 features)")
    print("  Platform : BITS Virtual Lab")
    print("=" * 95)

    # 1. Load test data
    test_csv = os.path.join(PROJECT_DIR, "test_data.csv")
    df = pd.read_csv(test_csv)
    print(f"\n[STEP 1] Loaded Test Dataset: {test_csv}")
    print(f"   - Total Test Samples  : {len(df):,}")
    print(f"   - Predictive Features : {len(df.columns) - 1}")
    print(f"   - Legitimate (0)      : {(df['is_fraud'] == 0).sum():,} transactions")
    print(f"   - Fraudulent (1)      : {(df['is_fraud'] == 1).sum():,} transactions")
    print(f"   - Class Imbalance     : 1:{(df['is_fraud'] == 0).sum() // (df['is_fraud'] == 1).sum()} (Severe Imbalance)")

    X_test = df.drop(columns=["is_fraud"]).values
    y_test = df["is_fraud"].values

    # 2. Evaluate all 5 models
    models = [
        ("Logistic Regression",     "logistic_regression.joblib", "Batch GD + Class Weights"),
        ("Decision Tree",           "decision_tree.joblib",       "CART + Gini Impurity"),
        ("KNN (k=7)",               "knn.joblib",                 "Euclidean + Batched Matrix Ops"),
        ("Naive Bayes (Gaussian)",  "naive_bayes.joblib",         "Gaussian Likelihood + Log-Space"),
        ("SVM (Linear SGD)",        "svm.joblib",                 "Hinge Loss + Sigmoid Prob Approx"),
    ]

    print("\n[STEP 2] Running Inference with 5 Pre-Trained Pure NumPy Scratch Models...")
    print("-" * 95)
    print(f"  {'Model':<26} {'Inference':>10} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}  {'Architecture'}")
    print("-" * 95)

    results = []
    for name, filename, arch in models:
        path = os.path.join(PROJECT_DIR, "model", filename)
        m = joblib.load(path)

        t0 = time.time()
        preds = m.predict(X_test)
        if hasattr(m, 'classes_') and m.classes_ is not None and 1 in m.classes_:
            pos_idx = int(np.where(m.classes_ == 1)[0][0])
        else:
            pos_idx = 1
        probs = m.predict_proba(X_test)[:, pos_idx]
        elapsed = (time.time() - t0) * 1000

        scores = MetricsScratch.compute_all(y_test, preds, probs)
        scores["Model"] = name
        results.append(scores)

        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(y_test, preds, positive_class=1)
        print(f"  {name:<26} {elapsed:8.1f}ms {TP:6d} {TN:6d} {FP:6d} {FN:6d}  {arch}")

    print("-" * 95)

    # 3. Print Final Comparison Table
    print("\n[STEP 3] Official Model Comparison Table (All 6 Required Metrics):")
    print("=" * 95)
    print(f"  {'ML Model Name':<28} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'MCC':>10}")
    print("=" * 95)
    winner_f1  = max(results, key=lambda x: x["F1"])
    winner_mcc = max(results, key=lambda x: x["MCC"])
    for r in results:
        tag = " * (Winner)" if r["Model"] == winner_f1["Model"] else ""
        disp_name = r["Model"] + tag
        print(f"  {disp_name:<28} {r['Accuracy']:>10.4f} {r['AUC']:>10.4f} {r['Precision']:>10.4f} {r['Recall']:>10.4f} {r['F1']:>10.4f} {r['MCC']:>10.4f}")
    print("=" * 95)
    if winner_f1["Model"] == winner_mcc["Model"]:
        print(f"  * Winner: {winner_f1['Model']} achieves both the highest balanced F1 Score ({winner_f1['F1']:.4f}) and highest MCC Score ({winner_mcc['MCC']:.4f}).")
    else:
        print(f"  * Winner: {winner_f1['Model']} achieves the highest F1 Score ({winner_f1['F1']:.4f}), while {winner_mcc['Model']} achieves the highest MCC Score ({winner_mcc['MCC']:.4f}).")

    # 4. Interactive App Launch Instructions
    print("\n[STEP 4] Interactive Streamlit Frontend Application:")
    print("   - Main App File : app.py")
    print("   - Features      : CSV Uploader, Model Dropdown, 6 Metric KPI Cards, Confusion Matrix Heatmap, ROC Curve")
    print("   - To run locally: streamlit run app.py")
    print("=" * 95)

if __name__ == "__main__":
    main()
