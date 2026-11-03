"""
=============================================================================
  K-NEAREST NEIGHBOR CLASSIFIER — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : KNN using Euclidean distance with batched prediction (no sklearn)
=============================================================================
"""

import numpy as np


class KNNScratch:
    """
    K-Nearest Neighbor Classifier implemented from scratch.

    Algorithm:
    ----------
    KNN is a lazy learner — fit() just stores training data.
    Prediction for each test sample:
      1. Compute Euclidean distance to every training sample.
      2. Find the k training samples with smallest distances.
      3. Take majority vote among k-nearest neighbours → predicted class.

    Euclidean Distance: d(a, b) = sqrt(Σ (a_i - b_i)²)

    Memory-efficient batched prediction is used to handle large datasets
    (avoids building an n_test × n_train distance matrix at once).

    Parameters
    ----------
    k          : int  — Number of nearest neighbours
    batch_size : int  — Number of test samples processed per batch
    """

    def __init__(self, k=5, batch_size=500):
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        self.k          = int(k)
        self.batch_size = int(batch_size)
        self.X_train    = None
        self.y_train    = None
        self.classes_   = None
        self.n_classes_ = 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Store training data with input validation and label mapping.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}")
        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # Map arbitrary class labels to contiguous indices 0 .. n_classes - 1
        y_mapped        = np.searchsorted(self.classes_, y)
        self.X_train    = np.asarray(X, dtype=np.float64)
        self.y_train    = y_mapped.astype(int)
        return self

    def _euclidean_distances_batch(self, X_batch):
        """
        Compute pairwise Euclidean distances between X_batch (test)
        and self.X_train (train) using vectorised numpy.

        ||a - b||² = ||a||² + ||b||² - 2 a·b^T

        Returns: shape (n_batch, n_train)
        """
        # sum of squares for each row
        sum_sq_test  = np.sum(X_batch  ** 2, axis=1, keepdims=True)  # (n_batch, 1)
        sum_sq_train = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # (1, n_train)
        dot_product  = X_batch.dot(self.X_train.T)                   # (n_batch, n_train)

        dist_sq = sum_sq_test + sum_sq_train - 2.0 * dot_product
        # Clip to avoid numerical negatives before sqrt
        return np.sqrt(np.maximum(dist_sq, 0))

    def _predict_batch(self, X_batch):
        """
        Predict classes and probabilities for a batch of test samples.
        Returns: (predictions, probabilities)  shapes (n_batch,), (n_batch, n_classes)
        """
        distances = self._euclidean_distances_batch(X_batch.astype(np.float64))

        n_train = len(self.X_train)
        actual_k = max(1, min(self.k, n_train))
        partition_idx = min(actual_k - 1, n_train - 1)

        # Indices of k nearest neighbours for each test sample
        k_idx = np.argpartition(distances, partition_idx, axis=1)[:, :actual_k]

        predictions = []
        probabilities = []

        for i in range(len(X_batch)):
            k_labels = self.y_train[k_idx[i]]

            # Majority vote
            counts = np.bincount(k_labels, minlength=self.n_classes_)
            pred   = int(np.argmax(counts))
            probs  = counts / actual_k

            predictions.append(pred)
            probabilities.append(probs)

        return np.array(predictions), np.array(probabilities)

    def predict(self, X):
        """
        Predict class labels for X.
        Returns np.ndarray of shape (n_samples,).
        """
        if len(X) == 0:
            return np.empty((0,), dtype=int)
        preds = []
        for start in range(0, len(X), self.batch_size):
            batch = X[start: start + self.batch_size]
            p, _  = self._predict_batch(batch)
            preds.append(p)
        raw_preds = np.concatenate(preds)
        if hasattr(self, 'classes_') and self.classes_ is not None:
            return self.classes_[raw_preds]
        return raw_preds

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        Returns np.ndarray of shape (n_samples, n_classes).
        """
        if len(X) == 0:
            return np.empty((0, self.n_classes_))
        probs = []
        for start in range(0, len(X), self.batch_size):
            batch = X[start: start + self.batch_size]
            _, p  = self._predict_batch(batch)
            probs.append(p)
        return np.concatenate(probs)
