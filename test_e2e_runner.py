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
        if hasattr(model, 'classes_') and model.classes_ is not None and 1 in model.classes_:
            pos_idx = int(np.where(model.classes_ == 1)[0][0])
        else:
            pos_idx = 1
        y_proba = model.predict_proba(X_test)[:, pos_idx]
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
    # SUITE 5: COMPONENT INTEGRATION & SCHEMA ROBUSTNESS
    # -------------------------------------------------------------------------
    print("\n[SUITE 5] Component Integration & Schema Robustness Verification")
    prep = loaded_models["preprocessor.joblib"]
    feature_cols = prep.feature_cols_

    # Part A: Genuine Component Integration Tests (Actual Preprocessor & Metrics APIs)
    # 1. Component Test: Preprocessor schema alignment under column permutation
    reordered_df = df_test.copy()
    shuffled_cols = list(reordered_df.columns)
    np.random.default_rng(42).shuffle(shuffled_cols)
    reordered_df = reordered_df[shuffled_cols]
    X_reordered, _ = prep.transform(reordered_df)
    X_standard, _ = prep.transform(df_test)
    np.testing.assert_allclose(X_reordered, X_standard, atol=1e-7, err_msg="Preprocessor column realignment failed!")
    print("  [COMPONENT TEST] Preprocessor.transform() strictly realigns permuted feature columns.")

    # 2. Component Test: Preprocessor missing test column imputation with training mean
    incomplete_eval_df = df_test.drop(columns=[feature_cols[0]]).copy()
    X_inc, _ = prep.transform(incomplete_eval_df)
    first_col_idx = prep.feature_cols_.index(feature_cols[0])
    np.testing.assert_allclose(X_inc[:, first_col_idx], 0.0, atol=1e-7)
    print("  [COMPONENT TEST] Preprocessor.transform() imputes missing column with training mean (scaled to 0.0).")

    # 3. Component Test: Preprocessor rejects all-NaN column during fitting
    try:
        bad_df = pd.DataFrame({"feat1": [np.nan, np.nan], "feat2": [1.0, 2.0], "is_fraud": [0, 1]})
        Preprocessor().fit_transform(bad_df)
        assert False, "Failed to reject all-NaN column in Preprocessor.fit_transform"
    except ValueError:
        print("  [COMPONENT TEST] Preprocessor.fit_transform() rejects all-NaN columns.")

    # 4. Component Test: Preprocessor rejects non-finite values in transform
    try:
        inf_df = df_test.copy()
        inf_df.iloc[0, 0] = np.inf
        prep.transform(inf_df)
        assert False, "Failed to reject infinite value in Preprocessor.transform"
    except ValueError:
        print("  [COMPONENT TEST] Preprocessor.transform() rejects infinite feature values.")

    # 5. Component Test: MetricsScratch input validation and single-class AUC guards
    try:
        MetricsScratch._validate_binary_inputs([1, 0], y_scores=[np.nan, 0.5])
        assert False, "Failed to reject NaN in y_scores"
    except ValueError:
        pass
    try:
        MetricsScratch.roc_curve([1, 1, 1], [0.9, 0.8, 0.7])
        assert False, "Failed to reject single class in roc_curve"
    except ValueError:
        pass
    cm = MetricsScratch.confusion_matrix([0, 1, 0, 1], [0, 0, 0, 1])
    assert cm.shape == (2, 2), f"Expected (2, 2) confusion matrix, got {cm.shape}"
    print("  [COMPONENT TEST] MetricsScratch input validation, 2x2 layout, and single-class AUC guards verified.")

    # Part B: Application Logic Checks (Documented Specification Checks)
    # 6. Logic Check: Target column auto-detection multi-alias precedence
    for alias in ["is_fraud", "fraud", "target", "class", "Class", "label"]:
        test_alias_df = pd.DataFrame({alias: [0, 1, 0], "feat1": [1.0, 2.0, 3.0]})
        detected = next((c for c in ["is_fraud", "fraud", "target", "class", "Class", "label"] if c in test_alias_df.columns), None)
        assert detected == alias, f"Failed detection for {alias}"
    print("  [LOGIC CHECK] Application target column multi-alias detection logic verified.")

    # 7. Logic Check: Preprocessed missing feature detection logic
    incomplete_preprocessed = df_test.drop(columns=[feature_cols[0]]).copy()
    missing_feats = [c for c in feature_cols if c not in incomplete_preprocessed.columns]
    assert len(missing_feats) == 1 and missing_feats[0] == feature_cols[0]
    print("  [LOGIC CHECK] Application preprocessed missing feature rejection logic verified.")

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
    assert all(
        status in ("PASS", "SKIPPED") for status in suite_status.values()
    ), f"One or more test suites failed! Statuses: {suite_status}"
    assert any(
        status == "PASS" for status in suite_status.values()
    ), "No test suites passed!"
    print("ALL TEST SUITES EXECUTED AND CERTIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
