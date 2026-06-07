<div align="center">

<img src="assets/demo.gif" alt="Stock Forecasting Dashboard Demo" width="800"/>

# 📈 Stock Price Forecasting

### Prophet vs LSTM — Comparative Time Series Forecasting with Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Prophet](https://img.shields.io/badge/Meta-Prophet-0467DF?style=flat)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Stock Price Forecasting** is a data science project that compares two powerful forecasting paradigms — Meta's **Prophet** (additive decomposition) vs. **LSTM** (sequence-to-sequence deep learning) — for predicting equity prices, with an interactive Streamlit dashboard for exploration.

[Live Demo](#) · [View Notebook](notebooks/) · [Report Bug](https://github.com/akash-dev-ai-brat/stock-forecasting/issues)

</div>

---

## 📌 Table of Contents
- [About the Project](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## 🎯 About the Project <a name="about"></a>

Time series forecasting is one of the most commercially impactful applications of ML — used in finance, supply chain, energy, and healthcare. This project does a rigorous head-to-head comparison of Prophet and LSTM on real stock price data (pulled via `yfinance`), measuring RMSE, MAE, and directional accuracy — then visualizes everything in an interactive dashboard.

The accompanying Jupyter notebook is a complete data science case study with EDA, stationarity tests, feature engineering, model training, and evaluation.

---

## ✨ Features <a name="features"></a>

- 🏦 **Live stock data** — fetch any ticker via Yahoo Finance API
- 🤖 **Dual model forecasting** — Prophet and LSTM predictions side by side
- 📊 **Error metrics dashboard** — RMSE, MAE, and MAPE displayed per model
- 📉 **Confidence interval visualization** — Prophet's uncertainty bands rendered on chart
- 🔢 **Custom forecast horizon** — adjust prediction window (7, 14, 30, 90 days)
- 📓 **Full EDA notebook** — decomposition, ACF/PACF plots, stationarity analysis

---

## 🛠️ Tech Stack <a name="tech-stack"></a>

| Layer | Technology |
|-------|-----------|
| Data Source | yfinance (Yahoo Finance) |
| Time Series Model 1 | Meta Prophet |
| Time Series Model 2 | LSTM (Keras / TensorFlow) |
| Data Analysis | Pandas, NumPy, Statsmodels |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

## 📊 Model Comparison <a name="model-comparison"></a>

Results on AAPL (30-day forecast, 2-year training window):

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| Prophet | 4.82 | 3.91 | 2.1% |
| LSTM | 3.67 | 2.84 | 1.6% |

> 📌 **Action item:** Replace with your actual results. Include the ticker, training window, and forecast period.

**Key finding:** LSTM outperforms Prophet on short-term precision; Prophet provides better interpretability and handles seasonality explicitly.

---

## 📁 Project Structure <a name="project-structure"></a>

```
stock-forecasting/
├── app.py                          # Streamlit dashboard — main entry point
├── src/
│   ├── data_loader.py              # yfinance data fetching & preprocessing
│   ├── prophet_model.py            # Prophet training & forecasting
│   ├── lstm_model.py               # LSTM architecture, training & inference
│   └── evaluator.py                # RMSE, MAE, MAPE computation
├── notebooks/
│   └── stock_forecasting_eda.ipynb # Full analysis notebook
├── assets/
│   └── demo.gif
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start <a name="quick-start"></a>

**1. Clone the repository**
```bash
git clone https://github.com/akash-dev-ai-brat/stock-forecasting.git
cd stock-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run app.py
```

Enter any stock ticker (e.g., `AAPL`, `TSLA`, `INFY`) and set your forecast horizon.

---

## ⚙️ How It Works <a name="how-it-works"></a>

```
yfinance API → Historical OHLCV Data
         │
         ├──→ Prophet Pipeline
         │       Trend + Seasonality Decomposition → Forecast
         │
         └──→ LSTM Pipeline
                 Normalize → Sliding Window Sequences → Train → Predict
                        │
                        ▼
              Streamlit Dashboard: Side-by-side plots + Error Metrics
```

---

## 📸 Screenshots <a name="screenshots"></a>



---

## 🔮 Future Improvements <a name="future-improvements"></a>

- [ ] Add Transformer-based (Informer) model for comparison
- [ ] Sentiment analysis integration using financial news headlines
- [ ] Portfolio-level forecasting (multi-stock)
- [ ] Backtesting engine with simulated trading performance

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/akash-dev-ai-brat">Akash Nath</a> · 
  <a href="https://www.linkedin.com/in/akash-nath-5aa816293/">LinkedIn</a>
</div>
