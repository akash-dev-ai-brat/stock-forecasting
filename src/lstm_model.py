import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


# ── 1. Data preparation ──────────────────────────────────────────────────────

def prepare_lstm_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract and clean Close prices from raw yfinance dataframe"""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    close = df[["Close"]].copy()
    close.index = pd.to_datetime(close.index)
    close.dropna(inplace=True)
    return close


def create_sequences(data: np.ndarray, lookback: int = 60):
    """
    Sliding window: for each position i, use last `lookback` days to predict day i
    Returns X of shape (samples, lookback, 1) and y of shape (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def split_and_scale(close_df: pd.DataFrame, test_ratio: float = 0.2, lookback: int = 60):
    """Scale data, split into train/test, create sequences"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_df.values)

    split_idx = int(len(scaled) * (1 - test_ratio))
    train_data = scaled[:split_idx]
    test_data  = scaled[split_idx - lookback:]   # include lookback window for test

    X_train, y_train = create_sequences(train_data, lookback)
    X_test,  y_test  = create_sequences(test_data,  lookback)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, split_idx


# ── 2. Model architecture ─────────────────────────────────────────────────────

def build_lstm_model(lookback: int = 60) -> Sequential:
    """2-layer stacked LSTM with dropout"""
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=32, activation="relu"),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


# ── 3. Training ───────────────────────────────────────────────────────────────

def train_lstm(model, X_train, y_train, epochs: int = 50, batch_size: int = 32):
    """Train with early stopping and learning rate reduction"""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate_lstm(model, X_test, y_test, scaler):
    """Predict, inverse transform, compute metrics"""
    predictions_scaled = model.predict(X_test)

    # Inverse transform back to original price scale
    predictions = scaler.inverse_transform(predictions_scaled)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae  = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    metrics = {
        "MAE" : round(float(mae),  4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4)
    }
    return predictions, actual, metrics