# Predicting Stock Price Movement Using Machine Learning

**ML Course Project — Spring 2025**  
Akash Dathaprasad · Bala Joseph Nithish Reddy Dondeti  
University of North Florida

---

## Overview

This project compares four machine learning models for predicting the next-day price direction (up or down) of large-cap US stocks. It is a binary classification task using daily OHLCV price data enriched with technical indicators and lagged return features.

**Models compared:**
- Logistic Regression (baseline ML model)
- Random Forest
- XGBoost
- LSTM (Long Short-Term Memory neural network)

---

## Project Structure

```
stock-price-prediction/
│
├── stock_prediction.py     # Main pipeline script (data → features → train → evaluate → plots)
├── stock_prediction.ipynb  # Jupyter notebook version (same pipeline, cell-by-cell)
├── requirements.txt        # Python dependencies
├── results_summary.csv     # Auto-generated results table after running the pipeline
├── figures/                # Auto-generated plots (created on first run)
│   ├── fig_accuracy_comparison.png
│   ├── fig_confusion_matrices.png
│   ├── fig_feature_importance.png
│   ├── fig_lstm_training.png
│   └── fig_cumulative_returns.png
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (script version)

```bash
python stock_prediction.py
```

This will:
- Download real stock data via `yfinance` (falls back to synthetic data if unavailable)
- Engineer all features
- Train all four models
- Print the results table
- Save all figures to the `figures/` folder
- Save `results_summary.csv`

To skip LSTM training (faster testing):
```bash
python stock_prediction.py --no-lstm
```

### 3. Run the notebook version

```bash
jupyter notebook stock_prediction.ipynb
```

Run all cells from top to bottom. Each cell is labelled with what it does.

---

## Features Used

| Feature | Description |
|---|---|
| MA5_20 | Ratio of 5-day to 20-day moving average |
| RSI | 14-day Relative Strength Index |
| MACD | 12-26 EMA difference |
| MACD_diff | MACD minus signal line |
| Vol_change | Day-over-day volume change |
| Return_1d | 1-day lagged log-return |
| Return_3d | 3-day lagged log-return |
| Return_5d | 5-day lagged log-return |
| Volatility | 10-day rolling std of daily log-returns |

---

## Dataset

- **Source:** Yahoo Finance via `yfinance`
- **Tickers:** AAPL, GOOGL, AMZN, MSFT, TSLA
- **Period:** January 2015 – December 2023
- **Train set:** 2015–2022 | **Test set:** 2023 (held-out)
- **Label:** 1 if next-day close > today's close, else 0

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
keras
matplotlib
seaborn
yfinance
```

See `requirements.txt` for pinned versions.

---

## Reproducibility Notes

- Random seed is fixed at `42` throughout.
- All features are computed strictly from past data (no lookahead).
- A 1-day gap between training and test set prevents data leakage across the split boundary.
- The LSTM model uses early stopping (patience=8) so results may vary slightly by hardware.
