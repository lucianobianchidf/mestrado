import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

def reset_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def to_binary_label(next_return):
    if np.isnan(next_return):
        return np.nan
    return 1 if next_return > 0 else 0

def validate_columns(df, required_cols, filename):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{filename}: colunas ausentes: {missing}")

def build_sequences(
    df,
    feature_cols,
    price_col,
    lookback,
    horizon,
):
    data = df.copy()

    #padroniza nomes
    data.columns = [str(col).strip() for col in data.columns]

    validate_columns(
        data,
        required_cols=["date", *feature_cols, price_col],
        filename="DataFrame",
    )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    numeric_cols = [*feature_cols, price_col]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = (
        data.sort_values("date")
        .dropna(subset=["date", *numeric_cols])
        .reset_index(drop=True)
    )

    prices = data[price_col].to_numpy(dtype=np.float64)

    future_returns = np.full(len(prices), np.nan, dtype=np.float64)
    if len(prices) > horizon:
        future_returns[:-horizon] = (
            prices[horizon:] / prices[:-horizon]
        ) - 1.0

    labels = np.array(
        [to_binary_label(ret) for ret in future_returns],
        dtype=np.float64,
    )

    feature_matrix = data[feature_cols].to_numpy(dtype=np.float32)

    X_list = []
    y_list = []

    last_valid_end = len(data) - horizon
    for end_idx in range(lookback - 1, last_valid_end):
        label = labels[end_idx]
        if np.isnan(label):
            continue

        start_idx = end_idx - lookback + 1
        X_list.append(feature_matrix[start_idx:end_idx + 1])
        y_list.append(int(label))

    if not X_list:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int32),
    )

def temporal_split(X, y, train_size, val_size):
    n_samples = len(X)
    train_end = int(n_samples * train_size)
    val_end = int(n_samples * (train_size + val_size))

    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )

def scale_3d_data(X_train, X_val, X_test):
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, n_features)
    val_2d = X_val.reshape(-1, n_features)
    test_2d = X_test.reshape(-1, n_features)

    scaler.fit(train_2d)

    X_train_scaled = scaler.transform(train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(val_2d).reshape(X_val.shape)
    X_test_scaled = scaler.transform(test_2d).reshape(X_test.shape)

    return (
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler,
    )

def compute_binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
