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
    Binary Logistic Regression implemented from scratch.

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
    learning_rate : float  — Step size for gradient descent
    n_iterations  : int    — Number of training epochs
    threshold     : float  — Decision boundary (default 0.5)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, threshold=0.5,
                 class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.threshold     = threshold
        self.class_weight  = class_weight   # None or 'balanced'
        self.weights       = None
        self.bias          = None
        self.loss_history  = []
        self._sample_weights = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sigmoid(self, z):
        """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})"""
        z_clipped = np.clip(z, -500, 500)   # prevent overflow
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _binary_cross_entropy(self, y_true, y_hat):
        """BCE loss = -1/n Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]"""
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_sample_weights(self, y):
        """
        Compute per-sample weights for class balancing.
        'balanced' mode: weight_c = n_samples / (n_classes * n_c)
        This gives higher weight to minority class samples.
        """
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        sample_w  = np.ones(n_samples)
        if self.class_weight == 'balanced':
            for c, cnt in zip(classes, counts):
                sample_w[y == c] = n_samples / (n_classes * cnt)
        return sample_w

    def fit(self, X, y, verbose=False):
        """
        Train the model using batch gradient descent.

        Parameters
        ----------
        X       : np.ndarray  shape (n_samples, n_features)
        y       : np.ndarray  shape (n_samples,)  — binary labels {0, 1}
        verbose : bool        print loss every 100 iterations
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        # Per-sample class weights (handles imbalance)
        sw = self._compute_sample_weights(y)   # shape (n_samples,)
        sw_sum = sw.sum()

        for i in range(self.n_iterations):
            # Forward pass
            z     = X.dot(self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # Weighted loss
            loss = self._binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            # Weighted gradients: dL/dw = (1/Sw) X^T (sw * (y_hat - y))
            error = sw * (y_hat - y)            # element-wise weight
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
        Output shape: (n_samples, 2)  → [P(class=0), P(class=1)]
        """
        z    = X.dot(self.weights) + self.bias
        prob = self._sigmoid(z)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        """
        Predicts class label using threshold.
        Returns np.ndarray of shape (n_samples,) with values {0, 1}
        """
        prob = self._sigmoid(X.dot(self.weights) + self.bias)
        return (prob >= self.threshold).astype(int)
