import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import download_stock_data
from prophet_model import (
    prepare_prophet_data, train_test_split_prophet,
    build_and_train_prophet, forecast_and_evaluate
)
from lstm_model import (
    prepare_lstm_data, split_and_scale,
    build_lstm_model, train_lstm, evaluate_lstm
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Price Forecasting")
st.markdown("Compare **Prophet** vs **LSTM** models on real stock data with proper evaluation metrics.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, RELIANCE.NS, INFY.NS"
    ).upper().strip()

    start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    end_date   = st.date_input("End Date",   value=pd.to_datetime("2024-12-31"))

    model_choice = st.selectbox(
        "Model",
        ["Both (Prophet + LSTM)", "Prophet only", "LSTM only"]
    )

    test_ratio = st.slider("Test set size", min_value=0.1, max_value=0.3,
                           value=0.2, step=0.05,
                           help="Fraction of data used for testing")

    lookback = st.slider("LSTM Lookback (days)", min_value=30, max_value=120,
                         value=60, step=10,
                         help="How many past days LSTM uses to predict next day")

    epochs = st.slider("LSTM Epochs", min_value=10, max_value=100,
                       value=50, step=10)

    run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    st.divider()
    st.caption("Built with Python · Prophet · TensorFlow · Streamlit")
    st.caption("Project by Akash")


# ── Helper: metric cards ──────────────────────────────────────────────────────
def metric_cards(label, metrics, color):
    st.markdown(f"#### {label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",    f"${metrics['MAE']:.2f}",  help="Mean Absolute Error")
    c2.metric("RMSE",   f"${metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    c3.metric("MAPE",   f"{metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error")


# ── Helper: forecast chart ────────────────────────────────────────────────────
def forecast_chart(train_dates, train_vals,
                   test_dates,  actual_vals,
                   pred_vals,   pred_label,
                   pred_color,  ci_upper=None, ci_lower=None, ticker=""):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_dates, y=train_vals,
        name="Train Data", line=dict(color="#aaaaaa", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=actual_vals,
        name="Actual Price", line=dict(color="#2196F3", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_vals,
        name=pred_label, line=dict(color=pred_color, width=2)
    ))

    if ci_upper is not None and ci_lower is not None:
        fig.add_trace(go.Scatter(
            x=list(test_dates) + list(test_dates)[::-1],
            y=list(ci_upper) + list(ci_lower)[::-1],
            fill="toself",
            fillcolor=f"rgba(255,152,0,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI"
        ))

    fig.update_layout(
        title=f"{ticker} — {pred_label} Forecast vs Actual",
        xaxis_title="Date", yaxis_title="Price",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


# ── Helper: residuals chart ───────────────────────────────────────────────────
def residuals_chart(test_dates, actual, predicted, label):
    residuals = np.array(actual) - np.array(predicted)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Residuals over Time", "Residuals Distribution"])

    fig.add_trace(go.Scatter(
        x=test_dates, y=residuals,
        line=dict(color="#9C27B0", width=1), name="Residual"
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="#9C27B0", name="Distribution"
    ), row=1, col=2)

    fig.update_layout(
        title=f"{label} — Residuals Analysis",
        height=350, template="plotly_white", showlegend=False
    )
    return fig


# ── Main logic ────────────────────────────────────────────────────────────────
if run_btn:
    # 1. Load data
    with st.spinner(f"Downloading {ticker} data from Yahoo Finance..."):
        try:
            df = download_stock_data(ticker,
                                     start=str(start_date),
                                     end=str(end_date))
            if df.empty:
                st.error(f"No data found for ticker **{ticker}**. Check the symbol and try again.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to download data: {e}")
            st.stop()

    st.success(f"✓ Loaded **{len(df):,}** trading days for **{ticker}**")

    # 2. Raw price chart
    st.subheader("📊 Historical Price")
    close_col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df.index, y=close_col,
        line=dict(color="#2196F3", width=1.5), name="Close Price"
    ))
    fig_raw.update_layout(height=300, template="plotly_white",
                          xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_raw, use_container_width=True)

    st.divider()

    # ── Prophet ──────────────────────────────────────────────────────────────
    run_prophet = model_choice in ["Both (Prophet + LSTM)", "Prophet only"]
    run_lstm    = model_choice in ["Both (Prophet + LSTM)", "LSTM only"]

    p_metrics, l_metrics = None, None

    if run_prophet:
        st.subheader("🔮 Prophet Model")
        with st.spinner("Training Prophet..."):
            prophet_df         = prepare_prophet_data(df)
            train_p, test_p    = train_test_split_prophet(prophet_df, test_ratio=test_ratio)
            p_model            = build_and_train_prophet(train_p)
            _, test_fc, p_metrics = forecast_and_evaluate(p_model, train_p, test_p)

        metric_cards("Prophet Metrics", p_metrics, "#FF9800")

        fig_p = forecast_chart(
            train_dates=train_p["ds"], train_vals=train_p["y"],
            test_dates=test_fc["ds"],  actual_vals=test_p["y"].values,
            pred_vals=test_fc["yhat"].values,
            pred_label="Prophet Forecast", pred_color="#FF9800",
            ci_upper=test_fc["yhat_upper"].values,
            ci_lower=test_fc["yhat_lower"].values,
            ticker=ticker
        )
        st.plotly_chart(fig_p, use_container_width=True)

        with st.expander("📉 Prophet Residuals"):
            st.plotly_chart(
                residuals_chart(test_fc["ds"], test_p["y"].values,
                                test_fc["yhat"].values, "Prophet"),
                use_container_width=True
            )

        st.divider()

    # ── LSTM ─────────────────────────────────────────────────────────────────
    if run_lstm:
        st.subheader("🧠 LSTM Model")
        with st.spinner("Training LSTM — this takes 1–2 minutes..."):
            close_df = prepare_lstm_data(df, ticker)
            X_train, y_train, X_test, y_test, scaler, split_idx = split_and_scale(
                close_df, test_ratio=test_ratio, lookback=lookback
            )
            l_model = build_lstm_model(lookback=lookback)
            history = train_lstm(l_model, X_train, y_train,
                                 epochs=epochs, batch_size=32)
            predictions, actual, l_metrics = evaluate_lstm(
                l_model, X_test, y_test, scaler
            )

        metric_cards("LSTM Metrics", l_metrics, "#E91E63")

        # Loss curve
        with st.expander("📉 Training Loss Curve"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history["loss"],
                name="Train Loss", line=dict(color="#2196F3", width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history["val_loss"],
                name="Val Loss", line=dict(color="#FF5722", width=2)
            ))
            fig_loss.update_layout(
                xaxis_title="Epoch", yaxis_title="MSE Loss",
                height=320, template="plotly_white"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        test_dates = close_df.index[split_idx:]
        fig_l = forecast_chart(
            train_dates=close_df.index[:split_idx],
            train_vals=close_df["Close"].values[:split_idx],
            test_dates=test_dates,
            actual_vals=actual.flatten(),
            pred_vals=predictions.flatten(),
            pred_label="LSTM Forecast", pred_color="#E91E63",
            ticker=ticker
        )
        st.plotly_chart(fig_l, use_container_width=True)

        with st.expander("📉 LSTM Residuals"):
            st.plotly_chart(
                residuals_chart(test_dates, actual.flatten(),
                                predictions.flatten(), "LSTM"),
                use_container_width=True
            )

        st.divider()

    # ── Comparison table ──────────────────────────────────────────────────────
    if p_metrics and l_metrics:
        st.subheader("⚔️ Prophet vs LSTM — Final Comparison")

        comp_df = pd.DataFrame({
            "Metric"  : ["MAE ($)", "RMSE ($)", "MAPE (%)"],
            "Prophet" : [p_metrics["MAE"],  p_metrics["RMSE"],  p_metrics["MAPE"]],
            "LSTM"    : [l_metrics["MAE"],  l_metrics["RMSE"],  l_metrics["MAPE"]],
        })
        comp_df["Winner 🏆"] = comp_df.apply(
            lambda r: "LSTM" if r["LSTM"] < r["Prophet"] else "Prophet", axis=1
        )

        st.dataframe(
            comp_df.style
                .highlight_min(subset=["Prophet", "LSTM"], color="#d4edda", axis=1)
                .format({"Prophet": "{:.4f}", "LSTM": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )

        # Bar chart comparison
        fig_comp = make_subplots(rows=1, cols=3,
            subplot_titles=["MAE", "RMSE", "MAPE (%)"])

        for i, metric in enumerate(["MAE ($)", "RMSE ($)", "MAPE (%)"]):
            row_data = comp_df[comp_df["Metric"] == metric].iloc[0]
            fig_comp.add_trace(go.Bar(
                x=["Prophet", "LSTM"],
                y=[row_data["Prophet"], row_data["LSTM"]],
                marker_color=["#FF9800", "#E91E63"],
                showlegend=False,
                text=[f"{row_data['Prophet']:.2f}", f"{row_data['LSTM']:.2f}"],
                textposition="outside"
            ), row=1, col=i+1)

        fig_comp.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True)

else:
    # Landing state
    st.info("👈 Configure settings in the sidebar and click **Run Forecast** to start.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("### 🔮 Prophet\nFacebook's time-series model. Fast, interpretable, handles seasonality well.")
    col2.markdown("### 🧠 LSTM\nDeep learning model. Captures complex non-linear patterns in price sequences.")
    col3.markdown("### 📊 Evaluation\nMAE, RMSE, MAPE metrics + residuals analysis. No cherry-picked graphs.")