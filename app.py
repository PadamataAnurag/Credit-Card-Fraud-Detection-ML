"""
=============================================================================
  BITS PILANI — WORK INTEGRATED LEARNING PROGRAMMES (WILP)
  Course    : Machine Learning (M.Tech AIML)
  Assignment: Assignment - 2
  Student   : PADAMATA ANURAG
  Roll No   : 2026ak05613
  Email     : 2026ak05613@wilp.bits-pilani.ac.in
  Platform  : BITS Virtual Lab / Streamlit Community Cloud
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
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

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Classifier | BITS Pilani",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .student-badge {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model Registry Mapping
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor (KNN)": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Sigmoid activation + Binary Cross-Entropy loss optimized via Batch Gradient Descent with class weights.",
    "Decision Tree": "CART algorithm using Gini Impurity split criterion with recursive depth-bounded binary partitioning.",
    "K-Nearest Neighbor (KNN)": "Vectorized Euclidean distance computing top-k majority vote with batched matrix algebra.",
    "Naive Bayes (Gaussian)": "Probabilistic Bayesian classifier computing Gaussian class-conditional feature likelihoods.",
    "Support Vector Machine (SVM)": "Linear SVM minimizing regularized Hinge Loss via Mini-Batch SGD with Platt scaling for probability estimates.",
}


# ---------------------------------------------------------------------------
# Model & Preprocessor Loader (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load preprocessor and all 5 pre-trained scratch models from model/."""
    model_dir = os.path.join(PROJECT_DIR, "model")
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Missing model file: {path}")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    return models, preprocessor


models_dict, preprocessor = load_all_models()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/1200px-BITS_Pilani-Logo.svg.png",
        width=120,
    )
    st.markdown("### 🎓 Academic Submission")
    st.markdown(
        """
        **Student**: PADAMATA ANURAG  
        **Roll No**: `2026ak05613`  
        **Email**: [2026ak05613@wilp.bits-pilani.ac.in](mailto:2026ak05613@wilp.bits-pilani.ac.in)  
        **Course**: Machine Learning  
        **Programme**: M.Tech (AIML)  
        **Division**: WILP, BITS Pilani  
        **Environment**: BITS Virtual Lab  
        """
    )
    st.divider()

    st.markdown("### ⚙️ Model Selection")
    selected_model_name = st.selectbox(
        "Choose Classification Model:",
        list(MODEL_FILES.keys()),
        index=3,  # Default to Naive Bayes (Overall winner)
    )

    st.caption(f"ℹ️ **Mechanism**: {MODEL_DESCRIPTIONS[selected_model_name]}")
    st.divider()

    st.markdown("### 📊 Dataset Details")
    st.markdown(
        """
        - **Dataset**: Credit Card Fraud 2026
        - **Total Instances**: 100,000
        - **Features**: 34 numerical features
        - **Target**: `is_fraud` (0: Legitimate, 1: Fraud)
        - **Evaluation Test Set**: 1,000 samples (Stratified)
        """
    )

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">💳 Credit Card Fraud Detection — ML Classifier Evaluator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="student-badge">
        <strong>PADAMATA ANURAG</strong> &nbsp;|&nbsp; Roll: <strong>2026ak05613</strong> &nbsp;|&nbsp; 
        Email: <strong>2026ak05613@wilp.bits-pilani.ac.in</strong> &nbsp;|&nbsp; 
        M.Tech (AIML) — BITS Pilani WILP (Assignment 2)
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Step 6(a): Dataset Upload (CSV)
# ---------------------------------------------------------------------------
st.subheader("📂 1. Test Dataset Upload (CSV)")

default_test_path = os.path.join(PROJECT_DIR, "test_data.csv")
uploaded_file = st.file_uploader(
    "Upload Test Data CSV (must include feature columns and 'is_fraud' target):",
    type=["csv"],
    help="Upload your test CSV. If no file is uploaded, the default pre-bundled stratified test dataset (1,000 instances) will be used.",
)

df_test = None
is_custom_upload = False

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        is_custom_upload = True
        st.success(f"✅ Successfully loaded uploaded CSV: `{uploaded_file.name}` ({df_test.shape[0]:,} rows, {df_test.shape[1]} columns)")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
elif os.path.exists(default_test_path):
    df_test = pd.read_csv(default_test_path)
    st.info(f"📁 Using bundled default test dataset: `test_data.csv` ({df_test.shape[0]:,} rows, {df_test.shape[1]} columns). You can upload a custom CSV above.")
else:
    st.warning("⚠️ Default `test_data.csv` not found. Please upload a test CSV file.")

if df_test is not None:
    # Auto-detect target column
    target_col = None
    for candidate in ["is_fraud", "fraud", "target", "class", "Class", "label"]:
        if candidate in df_test.columns:
            target_col = candidate
            break

    if target_col is None:
        st.warning("⚠️ Target column 'is_fraud' was not automatically detected.")
        target_col = st.selectbox("Please select the target column from your uploaded file:", list(df_test.columns))

    if target_col:
        # Preprocess / Extract X and y
        with st.expander("🔍 Preview Test Data & Class Distribution", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df_test.head(5), use_container_width=True)
            with col2:
                class_counts = df_test[target_col].value_counts().to_dict()
                st.write("**Class Counts:**")
                st.write(f"- Legitimate (0): `{class_counts.get(0, 0):,}`")
                st.write(f"- Fraudulent (1): `{class_counts.get(1, 0):,}`")
                fraud_rate = (class_counts.get(1, 0) / len(df_test)) * 100
                st.write(f"- Fraud Rate: `{fraud_rate:.2f}%`")

        # Feature separation & smart raw vs preprocessed detection
        is_raw_data = False
        if preprocessor is not None:
            has_str_cols = df_test.select_dtypes(include=['object', 'string', 'category']).shape[1] > 0
            has_id_cols = any(col in df_test.columns for col in getattr(preprocessor, 'ID_COLS', []) + getattr(preprocessor, 'LEAK_COLS', []))
            if has_str_cols or has_id_cols:
                is_raw_data = True

        if is_raw_data:
            try:
                X_eval, y_eval = preprocessor.transform(df_test)
            except Exception as ex:
                st.warning(f"Note: Raw transform encountered an issue ({ex}). Attempting direct numerical extraction...")
                X_df = df_test.drop(columns=[target_col]).select_dtypes(include=[np.number])
                X_eval = X_df.values.astype(float)
                y_eval = pd.to_numeric(df_test[target_col], errors='coerce').fillna(0).astype(int).values
        else:
            # Check if preprocessor feature columns exist and align them
            feature_cols = getattr(preprocessor, 'feature_cols_', None) if preprocessor is not None else None
            if feature_cols and all(col in df_test.columns for col in feature_cols):
                X_df = df_test[feature_cols].copy()
            else:
                drop_candidates = [target_col, "Unnamed: 0", "index", "id", "ID"]
                existing_drops = [c for c in drop_candidates if c in df_test.columns]
                X_df = df_test.drop(columns=existing_drops).select_dtypes(include=[np.number]).copy()

            X_eval = X_df.values.astype(float)
            y_eval = pd.to_numeric(df_test[target_col], errors='coerce').fillna(0).astype(int).values

        # Ensure no NaNs exist in X_eval
        if np.isnan(X_eval).any():
            col_means = np.nanmean(X_eval, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            nan_mask = np.isnan(X_eval)
            X_eval[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # -------------------------------------------------------------------
        # Step 6(b) & (c): Model Evaluation & Display 6 Evaluation Metrics
        # -------------------------------------------------------------------
        st.markdown(f"### 🤖 2. Model Evaluation — **{selected_model_name}**")

        active_model = models_dict[selected_model_name]

        # Compute predictions and probabilities
        with st.spinner(f"Computing predictions with {selected_model_name}..."):
            y_pred = active_model.predict(X_eval)
            y_proba = active_model.predict_proba(X_eval)[:, 1]
            metrics = MetricsScratch.compute_all(y_eval, y_pred, y_proba)

        # Display 6 Metric Cards
        st.markdown("#### 📈 Step 2 Required Evaluation Metrics (Calculated From Scratch):")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

        with m_col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Accuracy</div><div class="metric-value">{metrics["Accuracy"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">AUC Score</div><div class="metric-value">{metrics["AUC"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Precision</div><div class="metric-value">{metrics["Precision"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Recall</div><div class="metric-value">{metrics["Recall"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">F1 Score</div><div class="metric-value">{metrics["F1"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col6:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">MCC Score</div><div class="metric-value">{metrics["MCC"]:.4f}</div></div>',
                unsafe_allow_html=True,
            )

        st.caption("✨ Metrics are calculated strictly from mathematical first principles using pure NumPy (trapezoidal rule for AUC, harmonic mean for F1, Pearson phi for MCC).")

        # -------------------------------------------------------------------
        # Step 6(d): Confusion Matrix & Classification Breakdown
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🎯 3. Confusion Matrix & Diagnostic Visualizations")

        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(y_eval, y_pred, positive_class=1)
        cm = np.array([[TN, FP], [FN, TP]])

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("##### 📌 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred: Legit (0)", "Pred: Fraud (1)"],
                yticklabels=["Actual: Legit (0)", "Actual: Fraud (1)"],
                ax=ax,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
            ax.set_title(f"Confusion Matrix — {selected_model_name}", fontsize=12, pad=10)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(
                f"""
                - **True Negatives (TN)**: `{TN:,}` &nbsp;|&nbsp; **False Positives (FP)**: `{FP:,}`
                - **False Negatives (FN)**: `{FN:,}` &nbsp;|&nbsp; **True Positives (TP)**: `{TP:,}`
                """
            )

        with viz_col2:
            st.markdown("##### 📉 ROC Curve (Receiver Operating Characteristic)")
            fpr, tpr = MetricsScratch.roc_curve(y_eval, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            ax_roc.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {metrics['AUC']:.4f})")
            ax_roc.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Classifier")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
            ax_roc.set_title("ROC Curve Analysis", fontsize=12, pad=10)
            ax_roc.legend(loc="lower right", fontsize=9)
            ax_roc.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_roc)

        # Step 6(d): Detailed Classification Report
        with st.expander("📋 View Detailed Classification Report", expanded=True):
            p0 = TN / (TN + FN) if (TN + FN) > 0 else 0.0
            r0 = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0
            supp0 = TN + FP

            p1 = metrics["Precision"]
            r1 = metrics["Recall"]
            f1_1 = metrics["F1"]
            supp1 = TP + FN

            total_supp = supp0 + supp1
            macro_p = (p0 + p1) / 2
            macro_r = (r0 + r1) / 2
            macro_f1 = (f1_0 + f1_1) / 2

            weighted_p = (p0 * supp0 + p1 * supp1) / total_supp if total_supp > 0 else 0.0
            weighted_r = (r0 * supp0 + r1 * supp1) / total_supp if total_supp > 0 else 0.0
            weighted_f1 = (f1_0 * supp0 + f1_1 * supp1) / total_supp if total_supp > 0 else 0.0

            cr_df = pd.DataFrame([
                {"Category": "Class 0 (Legitimate)", "Precision": p0, "Recall": r0, "F1-Score": f1_0, "Support": supp0},
                {"Category": "Class 1 (Fraudulent)", "Precision": p1, "Recall": r1, "F1-Score": f1_1, "Support": supp1},
                {"Category": "Accuracy", "Precision": np.nan, "Recall": np.nan, "F1-Score": metrics["Accuracy"], "Support": total_supp},
                {"Category": "Macro Average", "Precision": macro_p, "Recall": macro_r, "F1-Score": macro_f1, "Support": total_supp},
                {"Category": "Weighted Average", "Precision": weighted_p, "Recall": weighted_r, "F1-Score": weighted_f1, "Support": total_supp},
            ])

            st.dataframe(
                cr_df.style.format({
                    "Precision": lambda v: f"{v:.4f}" if pd.notnull(v) else "",
                    "Recall": lambda v: f"{v:.4f}" if pd.notnull(v) else "",
                    "F1-Score": lambda v: f"{v:.4f}" if pd.notnull(v) else "",
                    "Support": "{:,}",
                }),
                use_container_width=True,
            )

        # -------------------------------------------------------------------
        # Step 5: Comparative Model Analysis
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🏆 4. Comprehensive All-Model Comparison")
        st.markdown("Benchmark performance across all **5 classification models** simultaneously on the current test data:")

        comparison_rows = []
        for idx, (m_name, m_inst) in enumerate(models_dict.items()):
            p = m_inst.predict(X_eval)
            pb = m_inst.predict_proba(X_eval)[:, 1]
            m_scores = MetricsScratch.compute_all(y_eval, p, pb)
            comparison_rows.append({
                "Model": m_name,
                "Accuracy": m_scores["Accuracy"],
                "AUC": m_scores["AUC"],
                "Precision": m_scores["Precision"],
                "Recall": m_scores["Recall"],
                "F1 Score": m_scores["F1"],
                "MCC Score": m_scores["MCC"],
            })

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comp_df.style.highlight_max(subset=["AUC", "F1 Score", "MCC Score"], color="#BBF7D0")
            .highlight_max(subset=["Accuracy", "Precision", "Recall"], color="#E0E7FF")
            .format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC Score": "{:.4f}",
            }),
            use_container_width=True,
        )
        st.info("💡 Highlighted cells indicate the highest metric score among the 5 models. **Naive Bayes** and **SVM** demonstrate strong AUC and balanced fraud recall under class imbalance.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
        <strong>BITS Pilani — Work Integrated Learning Programmes (WILP)</strong><br>
        Machine Learning Assignment - 2 | Student: <strong>PADAMATA ANURAG</strong> (Roll: <strong>2026ak05613</strong>)<br>
        Email: <code>2026ak05613@wilp.bits-pilani.ac.in</code> | Verified on BITS Virtual Lab
    </div>
    """,
    unsafe_allow_html=True,
)
