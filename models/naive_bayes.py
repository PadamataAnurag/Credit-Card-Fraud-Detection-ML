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
    Gaussian Naive Bayes implemented from scratch.

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

    Naive Bayes Assumption:
      Features are conditionally independent given the class label:
        P(x | Y=c) = Π_f P(x_f | Y=c)

    Parameters
    ----------
    var_smoothing : float — Fixed additive smoothing constant added to each feature variance
                            for numerical stability (prevents division by zero and log(0))
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_      = None
        self.priors_       = None   # log prior for each class
        self.means_        = None   # mean of each feature for each class
        self.variances_    = None   # variance of each feature for each class

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_gaussian(self, x, mean, var):
        """
        Log of Gaussian PDF:
        log N(x; μ, σ²) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]
        x   : shape (n_features,)
        mean: shape (n_features,)
        var : shape (n_features,)
        """
        log_two_pi_var = np.log(2.0 * np.pi * var)
        sq_diff        = (x - mean) ** 2 / var
        return -0.5 * np.sum(log_two_pi_var + sq_diff)

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
        n_samples, n_features = X.shape
        self.classes_         = np.unique(y)
        n_classes             = len(self.classes_)

        self.priors_   = np.zeros(n_classes)
        self.means_    = np.zeros((n_classes, n_features))
        self.variances_= np.zeros((n_classes, n_features))

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

    def _compute_log_posteriors(self, x):
        """Compute log-posterior for a single sample x."""
        log_posteriors = []
        for idx in range(len(self.classes_)):
            lp = self._log_priors[idx] + \
                 self._log_gaussian(x, self.means_[idx], self.variances_[idx])
            log_posteriors.append(lp)
        return np.array(log_posteriors)

    def predict(self, X):
        """
        Predict class labels using MAP (maximum a posteriori) estimate.
        Returns np.ndarray of shape (n_samples,).
        """
        results = []
        for x in X:
            log_posts = self._compute_log_posteriors(x)
            results.append(self.classes_[np.argmax(log_posts)])
        return np.array(results)

    def predict_proba(self, X):
        """
        Return normalised class probabilities using softmax over log-posteriors.
        Shape: (n_samples, n_classes).
        """
        proba = []
        for x in X:
            log_posts = self._compute_log_posteriors(x)

            # Softmax for numerical stability:  e^(log_p - max) / Σ e^(log_p - max)
            log_posts -= np.max(log_posts)
            exp_posts  = np.exp(log_posts)
            proba.append(exp_posts / exp_posts.sum())

        return np.array(proba)
