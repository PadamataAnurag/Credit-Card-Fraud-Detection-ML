"""
=============================================================================
  EVALUATION METRICS — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : All 6 required metrics computed from scratch using numpy only
=============================================================================

All 6 metrics required by assignment:
  1. Accuracy
  2. AUC Score          (Area Under ROC Curve — trapezoidal rule)
  3. Precision
  4. Recall
  5. F1 Score
  6. Matthews Correlation Coefficient (MCC)
"""

import numpy as np


class MetricsScratch:
    """
    Static class containing all evaluation metric implementations.
    No sklearn is used — all computed using numpy from first principles.
    """

    @staticmethod
    def _validate_binary_inputs(y_true, y_pred=None, y_scores=None):
        """
        Validate input dimensions, lengths, non-emptiness, and finite values.
        """
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or len(y_true) == 0:
            raise ValueError("y_true must be a non-empty 1D array.")
        if y_pred is not None:
            y_pred = np.asarray(y_pred)
            if y_pred.ndim != 1 or len(y_pred) != len(y_true):
                raise ValueError("y_pred must be 1D and match y_true length.")
        if y_scores is not None:
            y_scores = np.asarray(y_scores)
            if y_scores.ndim != 1 or len(y_scores) != len(y_true):
                raise ValueError("y_scores must be 1D and match y_true length.")
            if not np.all(np.isfinite(y_scores)):
                raise ValueError("y_scores contains NaN or non-finite values.")
        return y_true, y_pred, y_scores

    # ------------------------------------------------------------------
    # 1. Confusion Matrix Components
    # ------------------------------------------------------------------

    @staticmethod
    def confusion_matrix_components(y_true, y_pred, positive_class=1):
        """
        Compute TP, TN, FP, FN from binary predictions.

        TP = predicted positive AND actually positive
        TN = predicted negative AND actually negative
        FP = predicted positive AND actually negative  (Type I error)
        FN = predicted negative AND actually positive  (Type II error)
        """
        y_true, y_pred, _ = MetricsScratch._validate_binary_inputs(y_true, y_pred=y_pred)

        TP = int(np.sum((y_pred == positive_class) & (y_true == positive_class)))
        TN = int(np.sum((y_pred != positive_class) & (y_true != positive_class)))
        FP = int(np.sum((y_pred == positive_class) & (y_true != positive_class)))
        FN = int(np.sum((y_pred != positive_class) & (y_true == positive_class)))

        return TP, TN, FP, FN

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Returns 2×2 confusion matrix for binary classification:
            [[TN, FP],
             [FN, TP]]

        Layout:
            Row 0 (Actual Negative): [TN, FP]
            Row 1 (Actual Positive): [FN, TP]
        """
        y_true, y_pred, _ = MetricsScratch._validate_binary_inputs(y_true, y_pred=y_pred)
        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(y_true, y_pred, positive_class=1)
        return np.array([[TN, FP], [FN, TP]], dtype=int)

    # ------------------------------------------------------------------
    # 1. Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
                 = number of correct predictions / total predictions
        """
        y_true, y_pred, _ = MetricsScratch._validate_binary_inputs(y_true, y_pred=y_pred)
        return float(np.mean(y_true == y_pred))

    # ------------------------------------------------------------------
    # 2. AUC Score (Area Under the ROC Curve)
    # ------------------------------------------------------------------

    @staticmethod
    def roc_curve(y_true, y_scores, positive_class=1):
        """
        Compute ROC curve (FPR, TPR) by varying the decision threshold.

        FPR = FP / (FP + TN)  = False Positive Rate
        TPR = TP / (TP + FN)  = True Positive Rate = Recall
        """
        y_true, _, y_scores = MetricsScratch._validate_binary_inputs(y_true, y_scores=y_scores)
        y_true_binary = (y_true == positive_class).astype(int)

        n_pos = int(np.sum(y_true_binary == 1))
        n_neg = int(len(y_true_binary) - n_pos)

        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                "Both positive (class 1) and negative (class 0) samples are required to compute ROC AUC."
            )

        # Sort by score descending (highest score first)
        desc_idx = np.argsort(y_scores)[::-1]
        y_scores_sorted = y_scores[desc_idx]
        y_true_sorted = y_true_binary[desc_idx]

        # Group identical prediction scores (distinct threshold cutoffs)
        distinct_idx = np.where(np.diff(y_scores_sorted))[0]
        threshold_idxs = np.r_[distinct_idx, y_true_sorted.size - 1]

        tps = np.cumsum(y_true_sorted)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        tprs = np.r_[0.0, tps / n_pos]
        fprs = np.r_[0.0, fps / n_neg]

        return fprs, tprs

    @staticmethod
    def auc_score(y_true, y_scores, positive_class=1):
        """
        AUC = Area Under the ROC Curve computed via the Trapezoidal Rule:
              AUC = Σ (FPR_i+1 - FPR_i) * (TPR_i+1 + TPR_i) / 2
        """
        fprs, tprs = MetricsScratch.roc_curve(y_true, y_scores, positive_class)
        # Trapezoidal integration
        auc = float(np.trapz(tprs, fprs))
        return auc

    # ------------------------------------------------------------------
    # 3. Precision
    # ------------------------------------------------------------------

    @staticmethod
    def precision(y_true, y_pred, positive_class=1):
        """
        Precision = TP / (TP + FP)

        "Of all predicted positives, how many are actually positive?"
        """
        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(
            y_true, y_pred, positive_class)
        denom = TP + FP
        return float(TP / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Recall (Sensitivity / True Positive Rate)
    # ------------------------------------------------------------------

    @staticmethod
    def recall(y_true, y_pred, positive_class=1):
        """
        Recall = TP / (TP + FN)

        "Of all actual positives, how many were correctly identified?"
        """
        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(
            y_true, y_pred, positive_class)
        denom = TP + FN
        return float(TP / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # 5. F1 Score
    # ------------------------------------------------------------------

    @staticmethod
    def f1_score(y_true, y_pred, positive_class=1):
        """
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
           = 2*TP / (2*TP + FP + FN)

        Harmonic mean of Precision and Recall.
        """
        p = MetricsScratch.precision(y_true, y_pred, positive_class)
        r = MetricsScratch.recall(y_true, y_pred, positive_class)
        denom = p + r
        return float(2 * p * r / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # 6. Matthews Correlation Coefficient (MCC)
    # ------------------------------------------------------------------

    @staticmethod
    def mcc_score(y_true, y_pred, positive_class=1):
        """
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

        Range: [-1, +1]
          +1 = perfect prediction
           0 = random guessing
          -1 = completely inverted prediction

        MCC is reliable even with class imbalance (unlike accuracy).
        """
        TP, TN, FP, FN = MetricsScratch.confusion_matrix_components(
            y_true, y_pred, positive_class)

        numerator = float(TP * TN - FP * FN)
        denom_sq  = float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        if denom_sq == 0:
            return 0.0
        return numerator / np.sqrt(denom_sq)

    # ------------------------------------------------------------------
    # Convenience: compute all 6 metrics at once
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(cls, y_true, y_pred, y_scores, positive_class=1):
        """
        Compute all 6 metrics and return as a dict.

        Parameters
        ----------
        y_true        : true binary labels
        y_pred        : predicted binary labels
        y_scores      : predicted probability of positive class (for AUC)
        positive_class: which label is the "positive" class (default 1)
        """
        cls._validate_binary_inputs(y_true, y_pred=y_pred, y_scores=y_scores)
        return {
            "Accuracy" : round(cls.accuracy(y_true, y_pred), 6),
            "AUC"      : round(cls.auc_score(y_true, y_scores, positive_class), 6),
            "Precision": round(cls.precision(y_true, y_pred, positive_class), 6),
            "Recall"   : round(cls.recall(y_true, y_pred, positive_class), 6),
            "F1"       : round(cls.f1_score(y_true, y_pred, positive_class), 6),
            "MCC"      : round(cls.mcc_score(y_true, y_pred, positive_class), 6),
        }
