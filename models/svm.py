"""
=============================================================================
  SUPPORT VECTOR MACHINE (Linear) — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Linear SVM using Hinge Loss + Mini-Batch SGD (no sklearn)
=============================================================================
"""

import numpy as np


class SVMScratch:
    """
    Linear Support Vector Machine implemented from scratch.

    Algorithm (Primal SVM with SGD):
    ---------------------------------
    Objective (soft-margin, primal form):
        min_{w,b}  (1/2)||w||² + C * Σ max(0, 1 - y_i(w·x_i + b))
                   ─────────────   ──────────────────────────────
                  regularisation       hinge loss

    Gradients:
      If y_i(w·x_i + b) >= 1  (correctly classified with margin):
          dw = w,          db = 0
      Else (margin violation):
          dw = w - C*y_i*x_i,   db = -C*y_i

    Labels are internally mapped to {-1, +1} for SVM convention.

    For predict_proba (needed for AUC), class probabilities are estimated
    using a sigmoid-transformed margin probability approximation.

    Parameters
    ----------
    C            : float — Regularisation strength (larger C = less regularisation)
    learning_rate: float — SGD step size
    n_epochs     : int   — Number of passes over training data
    batch_size   : int   — Mini-batch size
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=50, batch_size=256,
                 class_weight=None):
        self.C             = C
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.class_weight  = class_weight   # None or 'balanced'
        self.weights       = None
        self.bias          = 0.0
        self.loss_history  = []
        self._class_weights_map = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_svm_labels(self, y):
        """Map {0,1} → {-1, +1} for SVM convention."""
        return np.where(y == 1, 1, -1).astype(float)

    def _hinge_loss(self, X, y_svm, sample_weight=None):
        """Compute hinge loss with optional sample weights (for monitoring)."""
        margins = y_svm * (X.dot(self.weights) + self.bias)
        hinge   = np.maximum(0, 1 - margins)
        reg     = 0.5 * np.dot(self.weights, self.weights)
        if sample_weight is not None:
            return float(reg + self.C * (np.sum(sample_weight * hinge) / np.sum(sample_weight)))
        return float(reg + self.C * np.mean(hinge))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, verbose=False):
        """
        Train using Mini-Batch Stochastic Gradient Descent.

        Parameters
        ----------
        X       : np.ndarray  shape (n_samples, n_features)
        y       : np.ndarray  shape (n_samples,)  labels {0, 1}
        verbose : bool        print loss every epoch
        """
        self.loss_history = []
        n_samples, n_features = X.shape
        y_svm = self._to_svm_labels(y)

        # Compute per-class weights for imbalance handling
        if self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            for c, cnt in zip(classes, counts):
                svm_c = 1 if c == 1 else -1
                self._class_weights_map[svm_c] = n_samples / (len(classes) * cnt)
        else:
            self._class_weights_map = {1: 1.0, -1: 1.0}

        # Full training set sample weights for epoch loss monitoring
        full_sample_w = np.array([self._class_weights_map.get(yi, 1.0) for yi in y_svm])

        # Initialise weights with small random values
        rng          = np.random.default_rng(seed=42)
        self.weights = rng.normal(0, 0.01, n_features)
        self.bias    = 0.0

        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            idx     = rng.permutation(n_samples)
            X_shuf  = X[idx]
            y_shuf  = y_svm[idx]

            for start in range(0, n_samples, self.batch_size):
                X_b = X_shuf[start: start + self.batch_size]
                y_b = y_shuf[start: start + self.batch_size]
                n_b = len(X_b)

                # Per-sample class weights for this batch
                sample_w = np.array([self._class_weights_map.get(yi, 1.0) for yi in y_b])
                sw_b_sum = sample_w.sum() if sample_w.sum() > 0 else float(n_b)

                # Margin: y_i * (w.x_i + b)
                margins = y_b * (X_b.dot(self.weights) + self.bias)

                # Mask for margin violations
                violation = margins < 1

                # Gradient of regularisation term (always)
                dw = self.weights.copy()
                db = 0.0

                # Weighted gradient of hinge loss (only violated samples)
                if violation.any():
                    sw_viol = sample_w[violation]
                    dw -= self.C * (sw_viol[:, None] * y_b[violation, None]
                                    * X_b[violation]).sum(axis=0) / sw_b_sum
                    db -= self.C * (sw_viol * y_b[violation]).sum() / sw_b_sum

                # Learning-rate decay
                lr = self.learning_rate / (1 + 0.01 * epoch)

                self.weights -= lr * dw
                self.bias    -= lr * db

            loss = self._hinge_loss(X, y_svm, sample_weight=full_sample_w)
            self.loss_history.append(loss)

            if verbose:
                print(f"  [SVM] Epoch {epoch+1:3d}/{self.n_epochs} | Hinge Loss: {loss:.6f}")

        return self

    def decision_function(self, X):
        """Returns raw decision scores (w·x + b). Shape: (n_samples,)"""
        return X.dot(self.weights) + self.bias

    def predict(self, X):
        """
        Predict class labels.
        decision > 0 → class 1 (Fraud), else class 0 (Legitimate).
        """
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def predict_proba(self, X):
        """
        Sigmoid-transformed probability approximation mapping raw margin distance to [0, 1].
        Returns shape (n_samples, 2).
        """
        scores = self.decision_function(X)
        # Sigmoid to convert scores to probabilities
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
        return np.column_stack([1.0 - prob_pos, prob_pos])
