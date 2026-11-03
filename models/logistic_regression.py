"""
=============================================================================
  LOGISTIC REGRESSION — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Binary Logistic Regression using Gradient Descent (no sklearn)
=============================================================================
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch using NumPy.

    Algorithm:
    ----------
    1. Initialize weights (w) and bias (b) to zeros.
    2. For each iteration:
       a. Compute linear combination:  z = X·w + b
       b. Apply sigmoid activation:    y_hat = sigmoid(z)
       c. Compute Binary Cross-Entropy loss.
       d. Compute gradients of loss w.r.t. w and b.
       e. Update:  w = w - lr * dw,   b = b - lr * db
    3. Use trained w and b to predict.

    Parameters
    ----------
    learning_rate : float  — Step size for gradient descent (must be positive finite number)
    n_iterations  : int    — Number of training epochs (must be positive integer)
    threshold     : float  — Decision boundary strictly in (0, 1) (default 0.5)
    class_weight  : str    — None or 'balanced'
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, threshold=0.5,
                 class_weight=None):
        if not (isinstance(learning_rate, (int, float, np.floating, np.integer)) and learning_rate > 0 and np.isfinite(learning_rate)):
            raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate}")
        if not (isinstance(n_iterations, (int, np.integer)) and n_iterations > 0):
            raise ValueError(f"n_iterations must be a positive integer, got {n_iterations}")
        if not (isinstance(threshold, (int, float, np.floating, np.integer)) and 0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be a float between 0 and 1, got {threshold}")
        if class_weight not in [None, 'balanced']:
            raise ValueError(f"class_weight must be None or 'balanced', got {class_weight}")

        self.learning_rate   = float(learning_rate)
        self.n_iterations    = int(n_iterations)
        self.threshold       = float(threshold)
        self.class_weight    = class_weight
        self.weights         = None
        self.bias            = 0.0
        self.loss_history    = []
        self._sample_weights = None
        self.classes_        = None
        self.n_features_in_  = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sigmoid(self, z):
        """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})"""
        z_clipped = np.clip(z, -500.0, 500.0)   # prevent numerical overflow
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _binary_cross_entropy(self, y_true, y_hat, sample_weight=None):
        """
        BCE loss = -1/n Σ w_i * [y_i · log(ŷ_i) + (1 - y_i) · log(1 - ŷ_i)]
        Supports sample weights for class-balanced training.
        """
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1.0 - eps)
        raw_loss = -(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))
        if sample_weight is not None:
            sw_sum = float(np.sum(sample_weight))
            return float(np.sum(sample_weight * raw_loss) / sw_sum) if sw_sum > 0 else float(np.mean(raw_loss))
        return float(np.mean(raw_loss))

    def _compute_sample_weights(self, y):
        """
        Compute per-sample weights for class balancing.
        'balanced' mode: weight_c = n_samples / (n_classes * n_c)
        """
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        sample_w  = np.ones(n_samples)
        if self.class_weight == 'balanced':
            for c, cnt in zip(classes, counts):
                sample_w[y == c] = n_samples / (n_classes * cnt)
        return sample_w

    def _validate_predict_input(self, X):
        """Validate input arrays prior to inference."""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        n_feat = getattr(self, 'n_features_in_', None)
        if n_feat is not None and X.shape[1] != n_feat:
            raise ValueError(f"Expected {n_feat} features, got {X.shape[1]}")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, verbose=False):
        """
        Train the model using batch gradient descent.

        Parameters
        ----------
        X       : np.ndarray  shape (n_samples, n_features)
        y       : np.ndarray  shape (n_samples,)  — binary labels
        verbose : bool        print loss every 100 iterations
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError(f"Sample count mismatch: len(X)={len(X)} != len(y)={len(y)}")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"LogisticRegression requires exactly 2 distinct classes, got {len(self.classes_)}: {self.classes_}")

        # Map binary classes to {0.0, 1.0} where class 1 is the positive class
        y_mapped = (y == self.classes_[1]).astype(float)

        self.loss_history = []
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        # Per-sample class weights (handles imbalance)
        sw = self._compute_sample_weights(y_mapped)
        sw_sum = float(sw.sum()) if sw.sum() > 0 else float(n_samples)

        for i in range(self.n_iterations):
            # Forward pass
            z     = X.dot(self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # Weighted loss
            loss = self._binary_cross_entropy(y_mapped, y_hat, sample_weight=sw)
            self.loss_history.append(loss)

            # Weighted gradients: dL/dw = (1/Sw) X^T (sw * (y_hat - y))
            error = sw * (y_hat - y_mapped)
            dw    = (1.0 / sw_sum) * X.T.dot(error)
            db    = (1.0 / sw_sum) * np.sum(error)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            if verbose and (i + 1) % 100 == 0:
                print(f"  [LR] Iter {i+1:4d}/{self.n_iterations} | Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):
        """
        Returns class probabilities.
        Output shape: (n_samples, 2).
        Column 0 represents self.classes_[0] (Negative).
        Column 1 represents self.classes_[1] (Positive).
        """
        X = self._validate_predict_input(X)
        if len(X) == 0:
            return np.empty((0, 2))

        z    = X.dot(self.weights) + self.bias
        prob = self._sigmoid(z)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X):
        """
        Predicts class label using threshold.
        Returns np.ndarray of shape (n_samples,) matching self.classes_.
        """
        X = self._validate_predict_input(X)
        if len(X) == 0:
            dtype = self.classes_.dtype if getattr(self, 'classes_', None) is not None else int
            return np.empty((0,), dtype=dtype)

        prob = self._sigmoid(X.dot(self.weights) + self.bias)
        pred_idx = (prob >= self.threshold).astype(int)
        if getattr(self, 'classes_', None) is not None:
            return self.classes_[pred_idx]
        return pred_idx
