import pandas as pd
import ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix yfinance MultiIndex columns (e.g. ("Close", "AAPL") → "Close")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # --- Trend indicators ---
    df["SMA_20"]  = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["EMA_20"]  = ta.trend.ema_indicator(df["Close"], window=20)

    # --- Momentum indicators ---
    df["RSI_14"]  = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"]   = macd.macd_diff()

    # --- Volatility indicators ---
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()

    # --- Volume indicator ---
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # --- Lag features ---
    for lag in [1, 3, 5, 10]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

    # --- Return features ---
    df["Daily_Return"]   = df["Close"].pct_change()
    df["Volatility_20d"] = df["Daily_Return"].rolling(20).std()

    # --- Calendar features ---
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month

    df.dropna(inplace=True)
    return df