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

    # ------------------------------------------------------------------
    # 1. Confusion Matrix Components
    # ------------------------------------------------------------------

    @staticmethod
    def confusion_matrix_components(y_true, y_pred, positive_class=1):
        """
        Compute TP, TN, FP, FN from raw predictions.

        TP = predicted positive AND actually positive
        TN = predicted negative AND actually negative
        FP = predicted positive AND actually negative  (Type I error)
        FN = predicted negative AND actually positive  (Type II error)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        TP = int(np.sum((y_pred == positive_class) & (y_true == positive_class)))
        TN = int(np.sum((y_pred != positive_class) & (y_true != positive_class)))
        FP = int(np.sum((y_pred == positive_class) & (y_true != positive_class)))
        FN = int(np.sum((y_pred != positive_class) & (y_true == positive_class)))

        return TP, TN, FP, FN

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Returns 2×2 confusion matrix:
            [[TN, FP],
             [FN, TP]]
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for t, p in zip(y_true, y_pred):
            cm[class_to_idx[t], class_to_idx[p]] += 1
        return cm

    # ------------------------------------------------------------------
    # 1. Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
                 = number of correct predictions / total predictions
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
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
        y_true   = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Sort by score descending (highest score first)
        sorted_idx = np.argsort(y_scores)[::-1]
        y_true_s   = y_true[sorted_idx]

        n_pos = np.sum(y_true == positive_class)
        n_neg = len(y_true) - n_pos

        tprs = [0.0]
        fprs = [0.0]
        tp = fp = 0

        for label in y_true_s:
            if label == positive_class:
                tp += 1
            else:
                fp += 1
            tprs.append(tp / n_pos if n_pos > 0 else 0)
            fprs.append(fp / n_neg if n_neg > 0 else 0)

        tprs.append(1.0)
        fprs.append(1.0)

        return np.array(fprs), np.array(tprs)

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
        return {
            "Accuracy" : round(cls.accuracy(y_true, y_pred), 6),
            "AUC"      : round(cls.auc_score(y_true, y_scores, positive_class), 6),
            "Precision": round(cls.precision(y_true, y_pred, positive_class), 6),
            "Recall"   : round(cls.recall(y_true, y_pred, positive_class), 6),
            "F1"       : round(cls.f1_score(y_true, y_pred, positive_class), 6),
            "MCC"      : round(cls.mcc_score(y_true, y_pred, positive_class), 6),
        }
