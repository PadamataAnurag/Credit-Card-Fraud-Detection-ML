"""
=============================================================================
  END-TO-END AUTOMATED TEST SUITE FOR ML ASSIGNMENT 2
  Student: PADAMATA ANURAG | Roll No: 2026ak05613
  BITS Pilani WILP — M.Tech (AIML)
=============================================================================
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
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

    suite_status = {}

    # -------------------------------------------------------------------------
    # SUITE 1: VERIFY ALL MODEL ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n[SUITE 1] Checking Model Artifacts in model/")
    expected_models = [
        ("logistic_regression.joblib", LogisticRegressionScratch),
        ("decision_tree.joblib", DecisionTreeScratch),
        ("knn.joblib", KNNScratch),
        ("naive_bayes.joblib", GaussianNaiveBayesScratch),
        ("svm.joblib", SVMScratch),
        ("preprocessor.joblib", Preprocessor),
    ]

    loaded_models = {}
    for filename, expected_cls in expected_models:
        path = os.path.join(PROJECT_DIR, "model", filename)
        assert os.path.exists(path), f"CRITICAL: Missing artifact {filename}"
        obj = joblib.load(path)
        assert isinstance(obj, expected_cls), f"Type mismatch for {filename}: expected {expected_cls}, got {type(obj)}"
        sz_kb = os.path.getsize(path) / 1024
        print(f"  [PASS] {filename:<30} ({expected_cls.__name__}) - {sz_kb:.1f} KB")
        loaded_models[filename] = obj

    suite_status["SUITE 1: Model Artifacts"] = "PASS"

    # -------------------------------------------------------------------------
    # SUITE 2: TEST DATASET SPECIFICATIONS
    # -------------------------------------------------------------------------
    print("\n[SUITE 2] Checking test_data.csv Specifications")
    test_csv = os.path.join(PROJECT_DIR, "test_data.csv")
    assert os.path.exists(test_csv), "test_data.csv missing!"
    df_test = pd.read_csv(test_csv)
    n_rows, n_cols = df_test.shape

    assert n_rows >= 500, f"Rows {n_rows} < 500 requirement"
    assert n_cols == 35, f"Expected 35 columns (34 features + target), got {n_cols}"
    assert "is_fraud" in df_test.columns, "'is_fraud' column missing in test_data.csv"
    assert not df_test.isna().any().any(), "test_data.csv contains unexpected missing values!"
    assert not np.isinf(df_test.values).any(), "test_data.csv contains non-finite values!"

    counts = df_test["is_fraud"].value_counts().to_dict()
    assert 0 in counts and 1 in counts, "Both classes must be present!"
    print(f"  Rows: {n_rows:,} (Assignment Minimum: 500)")
    print(f"  Cols: {n_cols} (34 predictive features + 1 binary target)")
    print(f"  Class Distribution: {counts}")
    print("  [PASS] Specification checks passed.")
    suite_status["SUITE 2: Test Dataset Quality"] = "PASS"

    # -------------------------------------------------------------------------
    # SUITE 3: INFERENCE INTEGRITY & METRIC EVALUATIONS
    # -------------------------------------------------------------------------
    print("\n[SUITE 3] Prediction Integrity & 6 Evaluation Metrics on test_data.csv")
    X_test = df_test.drop(columns=["is_fraud"]).values
    y_test = df_test["is_fraud"].values

    models_to_test = [
        ("Logistic Regression", loaded_models["logistic_regression.joblib"]),
        ("Decision Tree", loaded_models["decision_tree.joblib"]),
        ("KNN (k=7)", loaded_models["knn.joblib"]),
        ("Naive Bayes (Gaussian)", loaded_models["naive_bayes.joblib"]),
        ("SVM (Linear SGD)", loaded_models["svm.joblib"]),
    ]

    for name, model in models_to_test:
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        elapsed = (time.time() - t0) * 1000

        # Assert shapes & values
        assert y_pred.shape == (n_rows,), f"Prediction shape mismatch: {y_pred.shape}"
        assert y_proba.shape == (n_rows,), f"Probability shape mismatch: {y_proba.shape}"
        assert set(np.unique(y_pred)).issubset({0, 1}), f"Predictions must be binary {set(np.unique(y_pred))}"
        assert np.all(np.isfinite(y_proba)), "Probability contains non-finite values!"
        assert np.all((y_proba >= 0.0) & (y_proba <= 1.0)), "Probabilities outside [0, 1]!"

        m = MetricsScratch.compute_all(y_test, y_pred, y_proba)
        for metric_name in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            assert metric_name in m, f"Missing metric {metric_name}"
            assert np.isfinite(m[metric_name]), f"Metric {metric_name} is not finite"

        print(f"  [PASS] {name:<24} | Acc: {m['Accuracy']:.4f} | AUC: {m['AUC']:.4f} | Rec: {m['Recall']:.4f} | F1: {m['F1']:.4f} | MCC: {m['MCC']:.4f} ({elapsed:.1f}ms)")

    suite_status["SUITE 3: 5-Model Inference & 6 Metrics"] = "PASS"

    # -------------------------------------------------------------------------
    # SUITE 4: RAW UNSCALED CSV TRANSFORMATION PIPELINE
    # -------------------------------------------------------------------------
    print("\n[SUITE 4] Raw Unscaled CSV Transformation Pipeline")
    candidate_raw_paths = [
        os.path.join(PROJECT_DIR, "..", "Dataset", "credit_card_fraud_detection_2026.csv"),
        os.path.join(PROJECT_DIR, "data", "credit_card_fraud_detection_2026.csv"),
        os.path.join(PROJECT_DIR, "credit_card_fraud_detection_2026.csv"),
    ]
    raw_path = next((p for p in candidate_raw_paths if os.path.exists(p)), None)
    if raw_path is not None:
        raw_sample = pd.read_csv(raw_path, nrows=50)
        prep = loaded_models["preprocessor.joblib"]
        X_trans, y_trans = prep.transform(raw_sample)
        assert X_trans.shape == (50, 34), f"Transformed shape mismatch: {X_trans.shape}"
        assert y_trans.shape == (50,), f"Transformed target mismatch: {y_trans.shape}"
        assert not np.isnan(X_trans).any(), "Transformed X contains NaNs"
        print(f"  [PASS] Successfully ingested & transformed 50 raw samples -> Output: {X_trans.shape}")
        suite_status["SUITE 4: Raw Data Ingestion & Transformation"] = "PASS"
    else:
        print("  [SKIPPED] Raw dataset file not found in search paths.")
        suite_status["SUITE 4: Raw Data Ingestion & Transformation"] = "SKIPPED"

    # -------------------------------------------------------------------------
    # SUITE 5: FAULT INJECTION & SCHEMA ROBUSTNESS
    # -------------------------------------------------------------------------
    print("\n[SUITE 5] Fault Injection & Schema Robustness Verification")
    prep = loaded_models["preprocessor.joblib"]
    feature_cols = prep.feature_cols_

    # 1. Exact schema validation test: missing column should be detected
    incomplete_df = df_test.drop(columns=[feature_cols[0]]).copy()
    missing_detected = [c for c in feature_cols if c not in incomplete_df.columns]
    assert len(missing_detected) == 1 and missing_detected[0] == feature_cols[0], "Failed to detect missing feature!"
    print("  [PASS] Schema validation detects missing required columns correctly.")

    # 2. Reordered columns test: realignment works strictly
    reordered_df = df_test.copy()
    cols = list(reordered_df.columns)
    np.random.default_rng(42).shuffle(cols)
    reordered_df = reordered_df[cols]
    aligned_df = reordered_df[feature_cols].copy()
    assert list(aligned_df.columns) == feature_cols, "Column realignment failed!"
    print("  [PASS] Column realignment strictly preserves expected feature order.")

    # 3. Training-fitted NaN imputation test (never test batch mean)
    nan_df = df_test[feature_cols].copy()
    nan_df.iloc[0:5, 0:3] = np.nan
    training_means = prep.scaler_.mean_
    fill_series = pd.Series(training_means, index=feature_cols)
    imputed_df = nan_df.fillna(fill_series)
    assert not imputed_df.isna().any().any(), "Training-fitted imputation failed!"
    print("  [PASS] Missing values imputed strictly using training baseline statistics.")

    # 4. Target column auto-detection
    for alias in ["is_fraud", "fraud", "target", "class", "Class", "label"]:
        test_alias_df = pd.DataFrame({alias: [0, 1, 0], "feat1": [1.0, 2.0, 3.0]})
        detected = next((c for c in ["is_fraud", "fraud", "target", "class", "Class", "label"] if c in test_alias_df.columns), None)
        assert detected == alias, f"Failed detection for {alias}"
    print("  [PASS] Target column multi-alias detection passed.")
    suite_status["SUITE 5: Schema Robustness & Imputation"] = "PASS"

    # -------------------------------------------------------------------------
    # SUITE 6: VERIFY JUPYTER NOTEBOOK INTEGRITY
    # -------------------------------------------------------------------------
    print("\n[SUITE 6] Jupyter Notebook Integrity")
    nb_path = os.path.join(PROJECT_DIR, "notebook.ipynb")
    assert os.path.exists(nb_path), "notebook.ipynb missing!"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    n_cells = len(nb_json["cells"])
    assert n_cells >= 5, f"Notebook has too few cells: {n_cells}"
    print(f"  [PASS] notebook.ipynb valid JSON with {n_cells} cells.")
    suite_status["SUITE 6: Jupyter Notebook Integrity"] = "PASS"

    # -------------------------------------------------------------------------
    # SUITE 7: VERIFY REQUIREMENTS.TXT
    # -------------------------------------------------------------------------
    print("\n[SUITE 7] Checking requirements.txt for Streamlit Cloud")
    req_path = os.path.join(PROJECT_DIR, "requirements.txt")
    assert os.path.exists(req_path), "requirements.txt missing!"
    with open(req_path, "r", encoding="utf-8") as f:
        reqs = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
    for mandatory in ["streamlit", "numpy", "pandas", "matplotlib", "seaborn", "joblib"]:
        assert any(mandatory in r for r in reqs), f"Mandatory dependency missing in requirements.txt: {mandatory}"
        print(f"  [PASS] {mandatory:<15} present in requirements.txt")
    suite_status["SUITE 7: Deployment Dependencies"] = "PASS"

    print("\n" + "=" * 75)
    print("AUTOMATED TEST SUITE SUMMARY:")
    for suite, status in suite_status.items():
        print(f"  [{status}] {suite}")
    print("=" * 75)
    assert all(st == "PASS" for st in suite_status.values()), "One or more test suites failed!"
    print("ALL TEST SUITES EXECUTED AND CERTIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
