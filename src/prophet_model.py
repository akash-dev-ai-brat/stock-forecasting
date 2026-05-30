import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stock dataframe to Prophet format (ds, y)"""
    df = df.copy()

    # Fix MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    return prophet_df


def train_test_split_prophet(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split into 80% train, 20% test"""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Train end : {train['ds'].iloc[-1].date()}")
    print(f"Test start: {test['ds'].iloc[0].date()}")
    return train, test


def build_and_train_prophet(train: pd.DataFrame) -> Prophet:
    """Build and fit the Prophet model"""
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,   # controls trend flexibility
        seasonality_prior_scale=10.0,
        interval_width=0.95             # 95% confidence interval
    )
    model.fit(train)
    return model


def evaluate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, MAPE"""
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def forecast_and_evaluate(model: Prophet, train: pd.DataFrame, test: pd.DataFrame):
    """Generate forecast on test period and compute metrics"""
    # Combine train + test dates for forecast
    future = model.make_future_dataframe(periods=len(test), freq="B")  # B = business days
    forecast = model.predict(future)

    # Extract only test period predictions
    test_forecast = forecast.iloc[-len(test):][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    test_forecast = test_forecast.reset_index(drop=True)
    test = test.reset_index(drop=True)

    metrics = evaluate_metrics(test["y"], test_forecast["yhat"])
    return forecast, test_forecast, metrics