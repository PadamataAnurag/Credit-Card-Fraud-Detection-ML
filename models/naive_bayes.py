"""
=============================================================================
  GAUSSIAN NAIVE BAYES — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Naive Bayes with Gaussian likelihood (no sklearn)
=============================================================================
"""

import numpy as np


class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes implemented from scratch using NumPy.

    Algorithm (Bayes' Theorem + Gaussian Likelihood):
    -------------------------------------------------
    Training:
      For each class c:
        π_c  = P(Y = c) = n_c / n               (prior)
        μ_cf = mean of feature f for class c
        σ_cf = variance of feature f for class c

    Prediction:
      For a new sample x:
        log P(Y=c | x) ∝ log π_c + Σ_f log N(x_f; μ_cf, σ_cf)

      where log N(x; μ, σ²) = -0.5 * log(2πσ²) - (x-μ)² / (2σ²)

      Choose class with highest log-posterior.

    Parameters
    ----------
    var_smoothing : float — Fixed additive smoothing constant added to each feature variance
                            for numerical stability (prevents division by zero and log(0))
    """

    def __init__(self, var_smoothing=1e-9):
        if not (isinstance(var_smoothing, (int, float, np.floating, np.integer)) and var_smoothing > 0 and np.isfinite(var_smoothing)):
            raise ValueError(f"var_smoothing must be a positive finite number, got {var_smoothing}")

        self.var_smoothing   = float(var_smoothing)
        self.classes_        = None
        self.priors_         = None   # class priors
        self.means_          = None   # mean of each feature for each class
        self.variances_      = None   # variance of each feature for each class
        self._log_priors     = None
        self.n_features_in_  = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_gaussian(self, x, mean, var):
        """
        Log of Gaussian PDF:
        log N(x; μ, σ²) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]
        """
        log_two_pi_var = np.log(2.0 * np.pi * var)
        sq_diff        = (x - mean) ** 2 / var
        return -0.5 * np.sum(log_two_pi_var + sq_diff)

    def _validate_predict_input(self, X):
        """Validate input arrays and model state prior to inference."""
        if self.classes_ is None or self.means_ is None or self.variances_ is None:
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

    def _compute_log_posteriors(self, x):
        """Compute log-posterior for a single sample x."""
        log_posteriors = []
        for idx in range(len(self.classes_)):
            lp = self._log_priors[idx] + \
                 self._log_gaussian(x, self.means_[idx], self.variances_[idx])
            log_posteriors.append(lp)
        return np.array(log_posteriors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, verbose=False):
        """
        Estimate class priors, per-class feature means and variances.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)
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
        if X.shape[1] == 0:
            raise ValueError("X must contain at least 1 feature.")
        if any(v is None or str(v).strip().lower() in ['nan', 'none', 'null', ''] for v in y):
            raise ValueError("Target y contains missing or invalid label values.")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"Naive Bayes requires at least 2 distinct classes, got {n_classes}: {self.classes_}")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.priors_    = np.zeros(n_classes)
        self.means_     = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Prior:  π_c = n_c / n
            self.priors_[idx]    = len(X_c) / n_samples

            # Per-feature statistics
            self.means_[idx]     = np.mean(X_c, axis=0)
            self.variances_[idx] = np.var(X_c, axis=0) + self.var_smoothing

        # Store log priors for efficiency
        self._log_priors = np.log(self.priors_)

        if verbose:
            print(f"  [NB] Classes: {self.classes_}")
            print(f"  [NB] Priors : {self.priors_.round(4)}")

        return self

    def predict(self, X):
        """
        Predict class labels using MAP (maximum a posteriori) estimate.
        Returns np.ndarray of shape (n_samples,).
        """
        X = self._validate_predict_input(X)
        if len(X) == 0:
            return np.empty((0,), dtype=self.classes_.dtype)

        results = []
        for x in X:
            log_posts = self._compute_log_posteriors(x)
            results.append(self.classes_[np.argmax(log_posts)])
        return np.array(results)

    def predict_proba(self, X):
        """
        Return normalised class probabilities using softmax over log-posteriors.
        Shape: (n_samples, n_classes).
        Columns strictly match the order of self.classes_.
        """
        X = self._validate_predict_input(X)
        if len(X) == 0:
            return np.empty((0, len(self.classes_)))

        proba = []
        for x in X:
            log_posts = self._compute_log_posteriors(x)

            # Softmax for numerical stability:  e^(log_p - max) / Σ e^(log_p - max)
            log_posts -= np.max(log_posts)
            exp_posts  = np.exp(log_posts)
            proba.append(exp_posts / exp_posts.sum())

        return np.array(proba)
