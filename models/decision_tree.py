"""
=============================================================================
  DECISION TREE CLASSIFIER — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Binary CART Decision Tree using Gini Impurity (no sklearn)
=============================================================================
"""

import numpy as np


class _Node:
    """Internal decision tree node or leaf."""
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, class_probs=None):
        self.feature     = feature        # feature index to split on
        self.threshold   = threshold      # threshold value
        self.left        = left           # left child (_Node)
        self.right       = right          # right child (_Node)
        self.value       = value          # majority class (leaves only)
        self.class_probs = class_probs    # class probability array (leaves only)

    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """
    Binary CART Decision Tree implemented from scratch using NumPy.

    Algorithm (CART):
    -----------------
    1. At each node, find the feature & threshold that minimises
       weighted Gini impurity after the split.
    2. Recurse until a stopping criterion is met:
       - Depth >= max_depth
       - Samples < min_samples_split
       - Node is pure (only one class)
    3. Leaf node stores the majority class AND class proportions
       (needed for predict_proba / AUC).

    Gini impurity:  G = 1 - Σ p_i²
    Weighted child impurity: G_split = (n_L/n)*G_L + (n_R/n)*G_R
    Information gain = G_parent - G_split

    Parameters
    ----------
    max_depth         : int  — Maximum tree depth (must be positive integer)
    min_samples_split : int  — Min samples required to split a node (must be integer >= 2)
    n_thresholds      : int  — Number of candidate thresholds per feature (percentile-sampled
                               between 5th and 95th percentiles for efficient continuous partitioning)
    """

    def __init__(self, max_depth=10, min_samples_split=5, n_thresholds=20):
        if not (isinstance(max_depth, (int, np.integer)) and max_depth > 0):
            raise ValueError(f"max_depth must be a positive integer, got {max_depth}")
        if not (isinstance(min_samples_split, (int, np.integer)) and min_samples_split >= 2):
            raise ValueError(f"min_samples_split must be an integer >= 2, got {min_samples_split}")
        if not (isinstance(n_thresholds, (int, np.integer)) and n_thresholds > 0):
            raise ValueError(f"n_thresholds must be a positive integer, got {n_thresholds}")

        self.max_depth         = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.n_thresholds      = int(n_thresholds)
        self.root              = None
        self.classes_          = None
        self.n_classes_        = 2
        self.n_features_in_    = None

    # ------------------------------------------------------------------
    # Impurity & Splitting
    # ------------------------------------------------------------------

    def _gini(self, y):
        """Gini impurity G(y) = 1 - Σ p_i²"""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        probs  = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """
        Search over all features and candidate split thresholds (using percentile
        sampling between 5th and 95th percentiles for continuous feature efficiency).
        Returns (best_feature_idx, best_threshold) or (None, None).
        """
        n_samples, n_features = X.shape
        best_gain      = -np.inf
        best_feature   = None
        best_threshold = None
        parent_gini    = self._gini(y)

        for feature_idx in range(n_features):
            col = X[:, feature_idx]
            unique_vals = np.unique(col)

            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) <= self.n_thresholds:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            else:
                pcts = np.linspace(5, 95, self.n_thresholds)
                thresholds = np.percentile(col, pcts)
                thresholds = np.unique(thresholds)

            for t in thresholds:
                left_mask = col <= t
                n_left    = np.sum(left_mask)
                n_right   = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                gini_left  = self._gini(y[left_mask])
                gini_right = self._gini(y[~left_mask])
                gain = parent_gini - (n_left / n_samples) * gini_left - (n_right / n_samples) * gini_right

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature_idx
                    best_threshold = t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build CART binary tree."""
        n_samples = len(y)

        # Stopping condition 1: pure node
        if len(np.unique(y)) == 1:
            val = int(y[0])
            probs = np.zeros(self.n_classes_)
            probs[val] = 1.0
            return _Node(value=val, class_probs=probs)

        # Stopping condition 2: max depth or too few samples to split
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            counts      = np.bincount(y.astype(int), minlength=self.n_classes_)
            class_probs = counts / n_samples
            majority    = int(np.argmax(counts))
            return _Node(value=majority, class_probs=class_probs)

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:   # no informative split found
            counts      = np.bincount(y.astype(int), minlength=self.n_classes_)
            class_probs = counts / n_samples
            majority    = int(np.argmax(counts))
            return _Node(value=majority, class_probs=class_probs)

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature=feature, threshold=threshold,
                     left=left, right=right)

    def _traverse(self, x, node):
        """Traverse tree for a single sample, return leaf node."""
        if node.is_leaf():
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def _validate_predict_input(self, X):
        """Validate model state and prediction input array."""
        if self.root is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X)
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
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(f"Training data must contain at least 2 distinct classes, got {self.classes_}")

        self.n_features_in_ = X.shape[1]

        # Map arbitrary class labels to contiguous indices 0 .. n_classes - 1
        y_mapped = np.searchsorted(self.classes_, y)
        if verbose:
            print(f"  [DT] Building tree (max_depth={self.max_depth}) ...")
        self.root = self._build_tree(X, y_mapped, depth=0)
        if verbose:
            print("  [DT] Tree built successfully.")
        return self

    def predict(self, X):
        """Returns predicted class labels. Shape: (n_samples,)"""
        X = self._validate_predict_input(X)
        if len(X) == 0:
            dtype = self.classes_.dtype if getattr(self, 'classes_', None) is not None else int
            return np.empty((0,), dtype=dtype)

        raw_preds = np.array([self._traverse(x, self.root).value for x in X])
        if getattr(self, 'classes_', None) is not None:
            return self.classes_[raw_preds]
        return raw_preds

    def predict_proba(self, X):
        """
        Returns class probabilities from leaf node distributions.
        Shape: (n_samples, n_classes).
        Columns strictly match the order of self.classes_.
        """
        X = self._validate_predict_input(X)
        if len(X) == 0:
            return np.empty((0, self.n_classes_))

        proba = np.array([self._traverse(x, self.root).class_probs for x in X])
        return proba
