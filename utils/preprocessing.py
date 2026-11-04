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

    NaN Handling Policy:
    --------------------
    - Standalone fit(): computes nanmean/nanstd ignoring NaNs, rejecting columns that are all-NaN.
    - Standalone transform(): propagates NaNs naturally if present.
    - Integrated Preprocessor pipeline: all numeric missing values are strictly imputed using
      the training feature means prior to scaler transform, guaranteeing finite outputs.
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
        if X.shape[1] == 0:
            raise ValueError("X must contain at least 1 feature column.")
        if np.isinf(X).any():
            raise ValueError("StandardScalerScratch.fit() received infinite values (inf or -inf).")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0) if np.isnan(X).any() else np.mean(X, axis=0)
            self.std_  = np.nanstd(X, axis=0) if np.isnan(X).any() else np.std(X, axis=0)
        if not np.all(np.isfinite(self.mean_)):
            raise ValueError("StandardScalerScratch: Feature matrix contains a column with all NaN values.")
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScalerScratch must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least 1 feature column.")
        if X.shape[1] != len(self.mean_):
            raise ValueError(f"Expected {len(self.mean_)} features, got {X.shape[1]}")
        if np.isinf(X).any():
            raise ValueError("StandardScalerScratch.transform() received infinite values (inf or -inf).")
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
    5. Validate binary target labels
    6. Check for valid training columns (reject all-NaN columns and infs)
    7. Impute numeric NaNs with column mean
    8. Scale X with StandardScalerScratch (fit on train, transform on test)
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
        self.cat_cols_     = []      # feature names designated as categorical during training
        self.scaler_       = StandardScalerScratch()
        self.feature_cols_ = []      # final feature column names
        self.classes_      = None    # valid training target classes

    def _drop_columns(self, df):
        drop = self.LEAK_COLS + self.ID_COLS + self.HIGH_NULL_COLS
        existing = [c for c in drop if c in df.columns]
        return df.drop(columns=existing)

    def _encode_categoricals(self, df, fit=True):
        """
        Label-encode categorical columns with explicit missing-value imputation.
        Uses column mode learned during fit() to impute missing categorical entries.

        Categorical dtype consistency:
        - When fit=True: discovers categorical columns by dtype, encodes them, and records
          their names in self.cat_cols_.
        - When fit=False: applies the fitted encoder to every column in self.cat_cols_ (or self.encoders_),
          regardless of incoming DataFrame dtype (preventing numeric-inferred test categories from
          bypassing their encoder).
        """
        if fit:
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            cat_cols = [c for c in cat_cols if c != self.TARGET_COL]
            self.cat_cols_ = cat_cols
        else:
            cat_cols = getattr(self, "cat_cols_", list(self.encoders_.keys()))

        for col in cat_cols:
            if col not in df.columns:
                continue

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

    @staticmethod
    def _validate_binary_target(series, name="Target column"):
        """
        Strictly validate that series contains only integer-valued binary labels {0, 1}.
        Rejects fractional non-integers (e.g. 0.2, 1.2), non-standard integers (e.g. -1, 1),
        and missing or null values.
        """
        raw_vals = series.values
        if any(v is None or str(v).strip().lower() in ['nan', 'none', 'null', ''] for v in raw_vals):
            raise ValueError(f"{name} contains missing or null values.")

        try:
            numeric_vals = pd.to_numeric(series, errors='raise').values
        except (ValueError, TypeError) as ex:
            raise ValueError(f"{name} contains non-numeric values incompatible with binary labels {{0, 1}}: {ex}")

        # Check for fractional non-integers
        if not np.all(np.equal(numeric_vals, np.floor(numeric_vals))):
            frac_examples = numeric_vals[numeric_vals != np.floor(numeric_vals)][:3]
            raise ValueError(f"{name} contains fractional non-integer values (e.g. {frac_examples}). Expected strictly binary integers {{0, 1}}.")

        int_vals = numeric_vals.astype(int)
        unique_vals = set(np.unique(int_vals))
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"{name} contains invalid class labels {unique_vals}. Expected strictly {{0, 1}}.")

        return int_vals

    def fit_transform(self, df):
        """
        Full pipeline applied to TRAINING data.
        Returns: X_scaled (np.ndarray), y (np.ndarray)
        """
        if len(df) == 0:
            raise ValueError("Training DataFrame cannot be empty.")
        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COL}' not found in training DataFrame.")

        # Strict binary target validation
        y = self._validate_binary_target(df[self.TARGET_COL], name=f"Target column '{self.TARGET_COL}'")
        target_classes = sorted(set(np.unique(y)))
        if target_classes != [0, 1]:
            raise ValueError(f"Training target column '{self.TARGET_COL}' must contain both binary classes {{0, 1}}, got {target_classes}.")

        self.classes_ = [0, 1]

        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=True)

        X_df = df.drop(columns=[self.TARGET_COL])
        self.feature_cols_ = X_df.columns.tolist()
        if len(self.feature_cols_) == 0:
            raise ValueError("Feature matrix must contain at least 1 feature column.")

        X = X_df.values.astype(float)

        # Reject infinite values
        if np.isinf(X).any():
            raise ValueError("Training feature matrix contains infinite values (inf or -inf).")

        # Check for entirely missing training columns
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
        if not self.feature_cols_ or self.scaler_ is None or getattr(self.scaler_, 'mean_', None) is None:
            raise ValueError("Preprocessor must be fitted before transform.")
        if len(self.feature_cols_) == 0:
            raise ValueError("Feature matrix must contain at least 1 feature column.")

        df = df.copy()
        df = self._drop_columns(df)
        df = self._encode_categoricals(df, fit=False)

        has_target = self.TARGET_COL in df.columns
        if has_target:
            y = self._validate_binary_target(df[self.TARGET_COL], name=f"Test target column '{self.TARGET_COL}'")
            test_target_classes = sorted(set(np.unique(y)))
            train_classes = getattr(self, 'classes_', [0, 1])
            if train_classes is not None and not set(test_target_classes).issubset(set(train_classes)):
                raise ValueError(f"Test target classes {test_target_classes} do not match training target classes {train_classes}.")
            X_df = df.drop(columns=[self.TARGET_COL])
        else:
            y = None
            X_df = df

        # Align columns to training set; impute missing training features with the training mean
        for i, col in enumerate(self.feature_cols_):
            if col not in X_df.columns:
                X_df[col] = self.scaler_.mean_[i]
        X_df = X_df[self.feature_cols_]

        if len(X_df) == 0:
            return np.empty((0, len(self.feature_cols_))), (np.empty((0,), dtype=int) if has_target else None)

        X = X_df.values.astype(float)

        # Reject infinite values in inference data
        if np.isinf(X).any():
            raise ValueError("Inference feature matrix contains infinite values (inf or -inf).")

        # Impute NaNs using fitted training scaler means
        col_means = self.scaler_.mean_
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X_scaled = self.scaler_.transform(X)
        return X_scaled, y
