import yfinance as yf
import pandas as pd
import os

TICKERS = ["AAPL", "RELIANCE.NS", "INFY.NS"]

def download_stock_data(ticker: str, start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    df.dropna(inplace=True)
    df["Ticker"] = ticker
    return df

def load_all(save: bool = True) -> dict:
    all_data = {}
    for ticker in TICKERS:
        df = download_stock_data(ticker)
        if save:
            os.makedirs("data", exist_ok=True)
            df.to_csv(f"data/{ticker.replace('.', '_')}.csv")
        all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    data = load_all(save=True)
    for t, df in data.items():
        print(f"{t}: {df.shape[0]} rows | {df.index[0].date()} → {df.index[-1].date()}")