"""
=============================================================================
  DATA PREPROCESSOR — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Custom preprocessing pipeline for Credit Card Fraud dataset
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standard Scaler (from scratch)
# ---------------------------------------------------------------------------

class StandardScalerScratch:
    """
    Z-score normalisation from scratch.
    x_scaled = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if len(X) == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        self.mean_ = np.nanmean(X, axis=0) if np.isnan(X).any() else np.mean(X, axis=0)
        self.std_  = np.nanstd(X, axis=0) if np.isnan(X).any() else np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScalerScratch must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if X.shape[1] != len(self.mean_):
            raise ValueError(f"Expected {len(self.mean_)} features, got {X.shape[1]}")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Label Encoder (from scratch)
# ---------------------------------------------------------------------------

class LabelEncoderScratch:
    """
    Encodes categorical string/object labels into integers [0, K-1].

    Handling Unseen Categories & Nulls:
    -----------------------------------
    - Known categories are mapped to unique integer indices 0 <= i < K.
    - If an unseen category appears during inference (transform), it is assigned
      the dedicated unknown index `unknown_value` (-1), preserving numerical
      stability without halting execution.
    """

    def __init__(self, unknown_value=-1):
        self.classes_       = None
        self._class_map     = {}
        self.unknown_value  = int(unknown_value)

    def fit(self, arr):
        # Discard NaN/None when determining training categories
        valid_vals = [v for v in arr if pd.notna(v)]
        if len(valid_vals) == 0:
            raise ValueError("Categorical column contains no valid non-null values.")
        self.classes_   = sorted(set(valid_vals))
        self._class_map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, arr):
        unk = getattr(self, 'unknown_value', -1)
        return np.array([self._class_map.get(v, unk) for v in arr], dtype=int)

    def fit_transform(self, arr):
        return self.fit(arr).transform(arr)


# ---------------------------------------------------------------------------
# Main Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Full preprocessing pipeline for Credit Card Fraud Detection dataset.

    Steps (applied in this order):
    --------------------------------
    1. Drop leaky columns (fraud_type, fraud_amount_lost) and ID columns
    2. Drop browser column (>41% missing values)
    3. Explicit categorical missing value imputation (mode) and Label Encoding
    4. Separate features (X) and target (y = is_fraud)
    5. Check for valid training columns (reject all-NaN columns)
    6. Impute numeric NaNs with column mean
    7. Scale X with StandardScalerScratch (fit on train, transform on test)
    """

    # Columns that directly leak target information
    LEAK_COLS = ["fraud_type", "fraud_amount_lost"]

    # ID / datetime columns that carry no predictive value
    ID_COLS   = [
        "transaction_id", "customer_id", "card_id",
        "merchant_id", "transaction_datetime"
    ]

    # Columns to drop due to too many missing values
    HIGH_NULL_COLS = ["browser"]

    TARGET_COL = "is_fraud"

    def __init__(self):
        self.encoders_     = {}      # feature -> LabelEncoderScratch
        self.cat_modes_    = {}      # feature -> mode category (for missing value imputation)
        self.scaler_       = StandardScalerScratch()
        self.feature_cols_ = []      # final feature column names

    def _drop_columns(self, df):
        drop = self.LEAK_COLS + self.ID_COLS + self.HIGH_NULL_COLS
        existing = [c for c in drop if c in df.columns]
        return df.drop(columns=existing)

    def _encode_categoricals(self, df, fit=True):
        """
        Label-encode all object/string columns with explicit missing-value imputation.
        Uses column mode learned during fit() to impute missing categorical entries.
        """
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        # Remove target if it slipped in
        cat_cols = [c for c in cat_cols if c != self.TARGET_COL]

        for col in cat_cols:
            if fit:
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    raise ValueError(f"Categorical column '{col}' contains no valid values.")
                mode_val = non_null.mode()[0]
                self.cat_modes_[col] = mode_val
                # Impute missing values with training mode before encoding
                clean_series = df[col].fillna(mode_val).astype(str)
                enc = LabelEncoderScratch(unknown_value=-1)
                df[col] = enc.fit_transform(clean_series.values)
                self.encoders_[col] = enc
            else:
                cat_modes = getattr(self, 'cat_modes_', {})
                if col in self.encoders_:
                    mode_val = cat_modes.get(col, None)
                    clean_series = df[col].fillna(mode_val).astype(str) if mode_val is not None else df[col].astype(str)
                    df[col] = self.encoders_[col].transform(clean_series.values)
                else:
                    df[col] = 0   # unseen categorical column in test data

        return df

    def fit_transform(self, df):
        """
        Full pipeline applied to TRAINING data.
        Returns: X_scaled (np.ndarray), y (np.ndarray)
        """
        if len(df) == 0:
            raise ValueError("Training DataFrame cannot be empty.")
        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COL}' not found in training DataFrame.")

        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=True)

        # Separate target
        y = df[self.TARGET_COL].values.astype(int)
        X_df = df.drop(columns=[self.TARGET_COL])

        self.feature_cols_ = X_df.columns.tolist()

        X = X_df.values.astype(float)

        # Issue B check: Reject training data if any column has no valid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(X, axis=0)
        if not np.all(np.isfinite(col_means)):
            raise ValueError("Training data contains a feature column with no valid values.")

        # Impute any remaining NaNs with column mean
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Scale
        X_scaled = self.scaler_.fit_transform(X)
        return X_scaled, y

    def transform(self, df):
        """
        Pipeline applied to TEST / NEW data (uses fitted encoders & scaler).
        Returns: X_scaled (np.ndarray), y (np.ndarray) or None if no target
        """
        # Issue E check: Fitted guard
        if not self.feature_cols_ or self.scaler_ is None or getattr(self.scaler_, 'mean_', None) is None:
            raise ValueError("Preprocessor must be fitted before transform.")
        if len(df) == 0:
            return np.empty((0, len(self.feature_cols_))), (np.empty((0,), dtype=int) if self.TARGET_COL in df.columns else None)

        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=False)

        has_target = self.TARGET_COL in df.columns
        y = df[self.TARGET_COL].values.astype(int) if has_target else None

        if has_target:
            X_df = df.drop(columns=[self.TARGET_COL])
        else:
            X_df = df

        # Issue A correction: Impute missing training columns with the training mean
        for i, col in enumerate(self.feature_cols_):
            if col not in X_df.columns:
                X_df[col] = self.scaler_.mean_[i]
        X_df = X_df[self.feature_cols_]

        X = X_df.values.astype(float)

        # Impute NaNs using fitted training scaler means
        col_means = self.scaler_.mean_
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X_scaled = self.scaler_.transform(X)
        return X_scaled, y
