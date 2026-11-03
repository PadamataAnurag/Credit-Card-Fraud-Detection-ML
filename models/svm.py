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
    Linear Support Vector Machine implemented from scratch using NumPy.

    Algorithm (Primal Soft-Margin SVM with Mini-Batch SGD):
    -------------------------------------------------------
    Objective:
        min_{w,b}  (1/2)||w||² + C * Σ w_i * max(0, 1 - y_i(w·x_i + b))
                   ─────────────   ──────────────────────────────────────
                   regularisation           weighted hinge loss

    Labels are internally mapped to {-1, +1} for SVM convention.
    For predict_proba (needed for AUC), class probabilities are estimated
    using a sigmoid-transformed margin probability approximation.

    Parameters
    ----------
    C            : float — Regularisation strength (must be positive finite number)
    learning_rate: float — SGD step size (must be positive finite number)
    n_epochs     : int   — Number of passes over training data (must be positive integer)
    batch_size   : int   — Mini-batch size (must be positive integer)
    class_weight : str   — None or 'balanced'
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=50, batch_size=256,
                 class_weight=None):
        if not (isinstance(C, (int, float, np.floating, np.integer)) and C > 0 and np.isfinite(C)):
            raise ValueError(f"C must be a positive finite number, got {C}")
        if not (isinstance(learning_rate, (int, float, np.floating, np.integer)) and learning_rate > 0 and np.isfinite(learning_rate)):
            raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate}")
        if not (isinstance(n_epochs, (int, np.integer)) and n_epochs > 0):
            raise ValueError(f"n_epochs must be a positive integer, got {n_epochs}")
        if not (isinstance(batch_size, (int, np.integer)) and batch_size > 0):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if class_weight not in [None, 'balanced']:
            raise ValueError(f"class_weight must be None or 'balanced', got {class_weight}")

        self.C                  = float(C)
        self.learning_rate      = float(learning_rate)
        self.n_epochs           = int(n_epochs)
        self.batch_size         = int(batch_size)
        self.class_weight       = class_weight
        self.weights            = None
        self.bias               = 0.0
        self.loss_history       = []
        self._class_weights_map = {}
        self.classes_           = None
        self.n_features_in_     = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_svm_labels(self, y):
        """Map binary labels to {-1.0, +1.0} using self.classes_."""
        positive_label = self.classes_[1] if self.classes_ is not None and len(self.classes_) > 1 else 1
        return np.where(y == positive_label, 1.0, -1.0)

    def _hinge_loss(self, X, y_svm, sample_weight=None):
        """Compute regularized hinge loss with optional sample weights."""
        margins = y_svm * (X.dot(self.weights) + self.bias)
        hinge   = np.maximum(0.0, 1.0 - margins)
        reg     = 0.5 * np.dot(self.weights, self.weights)
        if sample_weight is not None:
            sw_sum = np.sum(sample_weight)
            return float(reg + self.C * (np.sum(sample_weight * hinge) / sw_sum))
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
        y       : np.ndarray  shape (n_samples,)  binary labels
        verbose : bool        print loss every epoch
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # 1. Input structural validation
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError(f"Sample count mismatch: len(X)={len(X)} != len(y)={len(y)}")
        if X.shape[1] < 1:
            raise ValueError("X must contain at least one feature column.")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")

        self.loss_history = []
        n_samples, n_features = X.shape
        self.n_features_in_   = n_features

        # 2. Class label normalization
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Binary SVM requires exactly 2 distinct classes, found {len(self.classes_)}: {self.classes_}")

        y_svm = self._to_svm_labels(y)

        # 3. Compute balanced class weights
        if self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            for c, cnt in zip(classes, counts):
                svm_c = 1.0 if c == self.classes_[1] else -1.0
                self._class_weights_map[svm_c] = float(n_samples / (2.0 * cnt))
        else:
            self._class_weights_map = {1.0: 1.0, -1.0: 1.0}

        # Full training set sample weights for epoch loss monitoring
        full_sample_w = np.array([self._class_weights_map.get(yi, 1.0) for yi in y_svm])

        # Initialise weights with small random values
        rng          = np.random.default_rng(seed=42)
        self.weights = rng.normal(0.0, 0.01, n_features)
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

                # Batch sample weights
                sample_w = np.array([self._class_weights_map.get(yi, 1.0) for yi in y_b])
                sw_b_sum = sample_w.sum() if sample_w.sum() > 0 else float(n_b)

                # Margin: y_i * (w·x_i + b)
                margins = y_b * (X_b.dot(self.weights) + self.bias)
                violation = margins < 1.0

                # Gradient of regularisation term
                dw = self.weights.copy()
                db = 0.0

                # Weighted gradient of hinge loss on margin violators
                if violation.any():
                    sw_viol = sample_w[violation]
                    dw -= self.C * (sw_viol[:, None] * y_b[violation, None] * X_b[violation]).sum(axis=0) / sw_b_sum
                    db -= self.C * (sw_viol * y_b[violation]).sum() / sw_b_sum

                # Learning-rate decay
                lr = self.learning_rate / (1.0 + 0.01 * epoch)
                self.weights -= lr * dw
                self.bias    -= lr * db

            # Record full-dataset loss (prevents mini-batch broadcasting mismatch)
            loss = self._hinge_loss(X, y_svm, sample_weight=full_sample_w)
            self.loss_history.append(loss)

            if verbose:
                print(f"  [SVM] Epoch {epoch+1:3d}/{self.n_epochs} | Hinge Loss: {loss:.6f}")

        return self

    def decision_function(self, X):
        """Returns raw decision margin scores (w·x + b)."""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if hasattr(self, 'n_features_in_') and self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")
        return X.dot(self.weights) + self.bias

    def predict(self, X):
        """Predict binary class labels corresponding to self.classes_."""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if len(X) == 0:
            dtype = self.classes_.dtype if getattr(self, 'classes_', None) is not None else int
            return np.empty((0,), dtype=dtype)
        scores = self.decision_function(X)
        pred_idx = (scores >= 0).astype(int)
        if getattr(self, 'classes_', None) is not None:
            return self.classes_[pred_idx]
        return pred_idx

    def predict_proba(self, X):
        """Sigmoid-transformed margin probability approximation."""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if len(X) == 0:
            return np.empty((0, 2))
        scores = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -500.0, 500.0)))
        return np.column_stack([1.0 - prob_pos, prob_pos])
