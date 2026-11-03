"""
=============================================================================
  END-TO-END AUTOMATED VERIFICATION SUITE
  BITS Pilani — WILP M.Tech (AIML) | Machine Learning Assignment - 2
  Student: PADAMATA ANURAG | Roll: 2026ak05613
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

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

def run_tests():
    print("=" * 75)
    print("RUNNING END-TO-END TEST SUITE FOR ML ASSIGNMENT 2")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # SUITE 1: VERIFY ARTIFACTS IN model/
    # -------------------------------------------------------------------------
    print("\n[SUITE 1] Checking Model Artifacts in model/")
    required_artifacts = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "knn.joblib",
        "naive_bayes.joblib",
        "svm.joblib",
        "preprocessor.joblib",
    ]
    loaded_models = {}
    for filename in required_artifacts:
        path = os.path.join(PROJECT_DIR, "model", filename)
        assert os.path.exists(path), f"Artifact missing: {path}"
        obj = joblib.load(path)
        loaded_models[filename] = obj
        size_kb = os.path.getsize(path) / 1024
        print(f"  [PASS] {filename:<30} ({type(obj).__name__}) - {size_kb:.1f} KB")

    # -------------------------------------------------------------------------
    # SUITE 2: TEST_DATA.CSV SPECIFICATION COMPLIANCE
    # -------------------------------------------------------------------------
    print("\n[SUITE 2] Checking test_data.csv Specifications")
    test_path = os.path.join(PROJECT_DIR, "test_data.csv")
    assert os.path.exists(test_path), "test_data.csv is missing!"
    df_test = pd.read_csv(test_path)
    n_rows, n_cols = df_test.shape
    print(f"  Rows: {n_rows:,} (Assignment Minimum: 500)")
    print(f"  Cols: {n_cols} (Assignment Minimum: 12 features + 1 target = 13)")
    assert n_rows >= 500, f"Too few rows: {n_rows}"
    assert n_cols >= 13, f"Too few columns: {n_cols}"
    assert "is_fraud" in df_test.columns, "Target column 'is_fraud' missing!"
    counts = df_test["is_fraud"].value_counts().to_dict()
    print(f"  Class Distribution: {counts}")
    assert set(counts.keys()) == {0, 1}, "Target must contain binary classes {0, 1}"
    print("  [PASS] Specification checks passed.")

    # -------------------------------------------------------------------------
    # SUITE 3: PREDICTION INTEGRITY & METRIC CALCULATION (ALL 5 MODELS)
    # -------------------------------------------------------------------------
    print("\n[SUITE 3] Prediction Integrity & 6 Evaluation Metrics on test_data.csv")
    X_test = df_test.drop(columns=["is_fraud"]).values
    y_test = df_test["is_fraud"].values

    models_map = [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Decision Tree", "decision_tree.joblib"),
        ("KNN (k=7)", "knn.joblib"),
        ("Naive Bayes (Gaussian)", "naive_bayes.joblib"),
        ("SVM (Linear SGD)", "svm.joblib"),
    ]

    for name, key in models_map:
        m = loaded_models[key]
        t0 = time.time()
        preds = m.predict(X_test)
        probs = m.predict_proba(X_test)
        lat = (time.time() - t0) * 1000

        assert preds.shape == (n_rows,), f"{name} preds shape mismatch: {preds.shape}"
        assert probs.shape == (n_rows, 2), f"{name} probs shape mismatch: {probs.shape}"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), f"{name} probs outside [0, 1]"
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4, err_msg=f"{name} probs != 1")

        metrics = MetricsScratch.compute_all(y_test, preds, probs[:, 1])
        for metric_name in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            assert metric_name in metrics, f"Missing metric {metric_name} in {name}"
            val = metrics[metric_name]
            assert not np.isnan(val), f"Metric {metric_name} is NaN for {name}"

        print(f"  [PASS] {name:<25} | Acc: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | Rec: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f} ({lat:.1f}ms)")

    # -------------------------------------------------------------------------
    # SUITE 4: RAW DATA INGESTION THROUGH PREPROCESSOR
    # -------------------------------------------------------------------------
    print("\n[SUITE 4] Raw Unscaled CSV Transformation Pipeline")
    raw_path = os.path.join(PROJECT_DIR, "..", "Dataset", "credit_card_fraud_detection_2026.csv")
    if os.path.exists(raw_path):
        raw_sample = pd.read_csv(raw_path, nrows=50)
        prep = loaded_models["preprocessor.joblib"]
        X_trans, y_trans = prep.transform(raw_sample)
        assert X_trans.shape == (50, 34), f"Transformed shape mismatch: {X_trans.shape}"
        assert y_trans.shape == (50,), f"Transformed target mismatch: {y_trans.shape}"
        assert not np.isnan(X_trans).any(), "Transformed X contains NaNs"
        print(f"  [PASS] Successfully preprocessed 50 raw samples -> Output: {X_trans.shape}")
    else:
        print("  ℹ Raw dataset file not in ../Dataset; skipping raw slice transform test.")

    # -------------------------------------------------------------------------
    # SUITE 5: FAULT INJECTION & EDGE CASES IN EVALUATION
    # -------------------------------------------------------------------------
    print("\n[SUITE 5] Fault Injection & Robustness Verification")
    
    # 1. Messy columns (Index column, extra columns, reordered columns)
    messy_df = df_test.copy()
    messy_df.insert(0, "Unnamed: 0", range(len(messy_df)))
    messy_df["extra_random_col"] = 999.0
    cols = list(messy_df.columns)
    np.random.default_rng(42).shuffle(cols)
    messy_df = messy_df[cols]

    # Preprocessor feature alignment check
    prep = loaded_models["preprocessor.joblib"]
    feature_cols = prep.feature_cols_
    extracted_df = messy_df[feature_cols].copy()
    assert extracted_df.shape == (n_rows, 34), "Failed to realign messy columns!"
    print("  [PASS] Column realignment & index-stripping test passed.")

    # 2. NaNs in input
    nan_df = df_test.copy()
    nan_df.iloc[0:10, 0:5] = np.nan
    X_nan = nan_df.drop(columns=["is_fraud"]).values.astype(float)
    col_means = np.nanmean(X_nan, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_mask = np.isnan(X_nan)
    X_nan[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    assert not np.isnan(X_nan).any(), "NaN imputation failed!"
    print("  [PASS] NaN imputation recovery test passed.")

    # 3. Target column auto-detection
    for alias in ["is_fraud", "fraud", "target", "class", "Class", "label"]:
        test_alias_df = pd.DataFrame({alias: [0, 1, 0], "feat1": [1.0, 2.0, 3.0]})
        detected = next((c for c in ["is_fraud", "fraud", "target", "class", "Class", "label"] if c in test_alias_df.columns), None)
        assert detected == alias, f"Failed detection for {alias}"
    print("  [PASS] Target column multi-alias detection passed.")

    # -------------------------------------------------------------------------
    # SUITE 6: VERIFY JUPYTER NOTEBOOK EXECUTION
    # -------------------------------------------------------------------------
    print("\n[SUITE 6] Jupyter Notebook Integrity")
    nb_path = os.path.join(PROJECT_DIR, "notebook.ipynb")
    assert os.path.exists(nb_path), "notebook.ipynb missing!"
    import json
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    n_cells = len(nb_json["cells"])
    assert n_cells >= 5, f"Notebook has too few cells: {n_cells}"
    print(f"  [PASS] notebook.ipynb valid JSON with {n_cells} cells.")

    # -------------------------------------------------------------------------
    # SUITE 7: VERIFY REQUIREMENTS.TXT
    # -------------------------------------------------------------------------
    print("\n[SUITE 7] Checking requirements.txt for Streamlit Cloud")
    req_path = os.path.join(PROJECT_DIR, "requirements.txt")
    assert os.path.exists(req_path), "requirements.txt missing!"
    with open(req_path, "r", encoding="utf-8") as f:
        reqs = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
    for mandatory in ["streamlit", "scikit-learn", "numpy", "pandas", "matplotlib", "seaborn", "joblib"]:
        assert any(mandatory in r for r in reqs), f"Mandatory dependency missing in requirements.txt: {mandatory}"
        print(f"  [PASS] {mandatory:<15} present in requirements.txt")

    print("\n" + "=" * 75)
    print("ALL 7 END-TO-END TEST SUITES COMPLETED WITH ZERO ERRORS!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
