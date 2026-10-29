"""
=============================================================================
  DATA PREPROCESSOR — Implemented From Scratch
  Author : PADAMATA ANURAG | Roll: 2026ak05613
  Course : Machine Learning — M.Tech (AIML), BITS Pilani WILP
  Desc   : Custom preprocessing pipeline for Credit Card Fraud dataset
=============================================================================
"""

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
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Label Encoder (from scratch)
# ---------------------------------------------------------------------------

class LabelEncoderScratch:
    """Encodes string/categorical labels as integers starting from 0."""

    def __init__(self):
        self.classes_    = None
        self._class_map  = {}

    def fit(self, arr):
        self.classes_   = sorted(set(arr))
        self._class_map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, arr):
        return np.array([self._class_map.get(v, -1) for v in arr])

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
    3. Encode all categorical (object) columns with LabelEncoderScratch
    4. Separate features (X) and target (y = is_fraud)
    5. Scale X with StandardScalerScratch (fit on train, transform on test)
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
        self.encoders_    = {}      # feature → LabelEncoderScratch
        self.scaler_      = StandardScalerScratch()
        self.feature_cols_= []      # final feature column names

    def _drop_columns(self, df):
        drop = self.LEAK_COLS + self.ID_COLS + self.HIGH_NULL_COLS
        existing = [c for c in drop if c in df.columns]
        return df.drop(columns=existing)

    def _encode_categoricals(self, df, fit=True):
        """Label-encode all object/string columns."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove target if it slipped in
        cat_cols = [c for c in cat_cols if c != self.TARGET_COL]

        for col in cat_cols:
            if fit:
                enc = LabelEncoderScratch()
                df[col] = enc.fit_transform(df[col].astype(str).values)
                self.encoders_[col] = enc
            else:
                if col in self.encoders_:
                    df[col] = self.encoders_[col].transform(df[col].astype(str).values)
                else:
                    df[col] = 0   # unseen column in test

        return df

    def fit_transform(self, df):
        """
        Full pipeline applied to TRAINING data.
        Returns: X_scaled (np.ndarray), y (np.ndarray)
        """
        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=True)

        # Separate target
        y = df[self.TARGET_COL].values.astype(int)
        X_df = df.drop(columns=[self.TARGET_COL])

        self.feature_cols_ = X_df.columns.tolist()

        X = X_df.values.astype(float)

        # Handle any remaining NaNs (fill with column mean)
        col_means = np.nanmean(X, axis=0)
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
        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=False)

        has_target = self.TARGET_COL in df.columns
        y = df[self.TARGET_COL].values.astype(int) if has_target else None

        if has_target:
            X_df = df.drop(columns=[self.TARGET_COL])
        else:
            X_df = df

        # Align columns to training set
        for col in self.feature_cols_:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[self.feature_cols_]

        X = X_df.values.astype(float)

        # Handle NaNs
        col_means = self.scaler_.mean_
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X_scaled = self.scaler_.transform(X)
        return X_scaled, y
