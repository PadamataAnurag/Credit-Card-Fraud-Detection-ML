"""
=============================================================================
  DECISION TREE CLASSIFIER — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : CART Decision Tree using Gini Impurity (no sklearn)
=============================================================================
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal Node structure
# ---------------------------------------------------------------------------

class _Node:
    """Represents a single node in the decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None,
                 class_probs=None):
        self.feature     = feature      # Feature index used for splitting
        self.threshold   = threshold    # Threshold value for split
        self.left        = left         # Left child node  (X[:,f] <= threshold)
        self.right       = right        # Right child node (X[:,f] >  threshold)
        self.value       = value        # Majority class (for leaf nodes)
        self.class_probs = class_probs  # Class probabilities (for predict_proba)

    def is_leaf(self):
        return self.value is not None


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class DecisionTreeScratch:
    """
    Binary CART Decision Tree implemented from scratch.

    Algorithm (CART):
    -----------------
    1. At each node, find the feature & threshold that minimises
       weighted Gini impurity after the split.
    2. Recurse until a stopping criterion is met:
       - Depth ≥ max_depth
       - Samples < min_samples_split
       - Node is pure (only one class)
    3. Leaf node stores the majority class AND class proportions
       (needed for predict_proba / AUC).

    Gini impurity:  G = 1 - Σ p_i²
    Weighted child impurity: G_split = (n_L/n)*G_L + (n_R/n)*G_R
    Information gain = G_parent - G_split

    Parameters
    ----------
    max_depth         : int  — Maximum tree depth
    min_samples_split : int  — Min samples required to split a node
    n_thresholds      : int  — Number of candidate thresholds per feature (percentile-sampled
                               between 5th and 95th percentiles for efficient continuous partitioning)
    """

    def __init__(self, max_depth=10, min_samples_split=5, n_thresholds=20):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_thresholds      = n_thresholds
        self.root              = None
        self.n_classes_        = 2

    # ------------------------------------------------------------------
    # Impurity
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

            # Sample candidate thresholds from percentiles to speed up
            unique_vals = np.unique(col)
            if len(unique_vals) <= self.n_thresholds:
                thresholds = unique_vals
            else:
                percentiles = np.linspace(5, 95, self.n_thresholds)
                thresholds  = np.percentile(col, percentiles)
                thresholds  = np.unique(thresholds)

            for thr in thresholds:
                left_mask  = col <= thr
                right_mask = ~left_mask

                n_l = left_mask.sum()
                n_r = right_mask.sum()
                if n_l == 0 or n_r == 0:
                    continue

                # Weighted Gini after split
                gini_split = (n_l / n_samples) * self._gini(y[left_mask]) + \
                             (n_r / n_samples) * self._gini(y[right_mask])

                gain = parent_gini - gini_split

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature_idx
                    best_threshold = thr

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Tree Building (recursive)
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes  = len(np.unique(y))

        # ----- Stopping criteria -----
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            counts      = np.bincount(y.astype(int), minlength=self.n_classes_)
            class_probs = counts / n_samples
            majority    = int(np.argmax(counts))
            return _Node(value=majority, class_probs=class_probs)

        # ----- Find best split -----
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, verbose=False):
        self.n_classes_ = len(np.unique(y))
        if verbose:
            print(f"  [DT] Building tree (max_depth={self.max_depth}) ...")
        self.root = self._build_tree(X, y.astype(int), depth=0)
        if verbose:
            print("  [DT] Tree built successfully.")
        return self

    def _traverse(self, x, node):
        """Traverse tree for a single sample, return leaf node."""
        if node.is_leaf():
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        """Returns predicted class labels. Shape: (n_samples,)"""
        return np.array([self._traverse(x, self.root).value for x in X])

    def predict_proba(self, X):
        """
        Returns class probabilities from leaf node distributions.
        Shape: (n_samples, n_classes)
        """
        proba = []
        for x in X:
            leaf = self._traverse(x, self.root)
            proba.append(leaf.class_probs)
        return np.array(proba)
