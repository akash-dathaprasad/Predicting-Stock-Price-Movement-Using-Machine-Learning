"""
stock_prediction.py
====================
Predicting Stock Price Movement Using Machine Learning
ML Course Project — Spring 2025

Authors : Akash Dathaprasad, Bala Joseph Nithish Reddy Dondeti
Dataset : Daily OHLCV data via yfinance (2015-2023)
          

Usage:
    python stock_prediction.py            # run full pipeline
    python stock_prediction.py --no-lstm  # skip LSTM (faster testing)
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── Configuration ────────────────────────────────────────────────────────────
TICKERS     = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
START_DATE  = '2015-01-01'
END_DATE    = '2023-12-31'
TEST_YEAR   = 2023           # hold-out year
SEQ_LEN     = 20             # look-back window for LSTM
FIGURES_DIR = 'figures'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_yfinance(tickers, start, end):
    """Download real data using yfinance."""
    try:
        import yfinance as yf
        frames = []
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            df['Ticker'] = ticker
            frames.append(df)
        data = pd.concat(frames)
        data.index.name = 'Date'
        print(f"[INFO] Downloaded real data: {len(data)} rows across {len(tickers)} tickers.")
        return data
    except Exception as e:
        print(f"[WARN] yfinance failed ({e}). Using synthetic data instead.")
        return None


def generate_synthetic_data(tickers, start, end, seed=42):
    """
    Simulate daily OHLCV data using Geometric Brownian Motion.
    Produces realistic-looking price series for reproducible demo runs.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)   # business days only
    frames = []

    params = {   # (mu_annual, sigma_annual, S0)
        'AAPL' : (0.25, 0.28, 27.0),
        'GOOGL': (0.20, 0.26, 35.0),
        'AMZN' : (0.30, 0.35, 15.0),
        'MSFT' : (0.22, 0.24, 46.0),
        'TSLA' : (0.45, 0.65, 12.0),
    }

    for ticker in tickers:
        mu, sigma, S0 = params.get(ticker, (0.20, 0.30, 50.0))
        n   = len(dates)
        dt  = 1 / 252
        ret = rng.normal((mu - 0.5 * sigma**2) * dt,
                          sigma * np.sqrt(dt), n)
        price = S0 * np.exp(np.cumsum(ret))

        # Construct OHLCV from close price
        noise = lambda scale: rng.uniform(-scale, scale, n) * price
        high   = price + np.abs(noise(0.012))
        low    = price - np.abs(noise(0.012))
        open_  = price + noise(0.006)
        volume = rng.integers(5_000_000, 50_000_000, n).astype(float)

        df = pd.DataFrame({
            'Open'  : open_,
            'High'  : high,
            'Low'   : low,
            'Close' : price,
            'Volume': volume,
            'Ticker': ticker,
        }, index=dates)
        df.index.name = 'Date'
        frames.append(df)

    data = pd.concat(frames)
    print(f"[INFO] Generated synthetic data: {len(data)} rows across {len(tickers)} tickers.")
    return data


# ─── 2. Feature Engineering ───────────────────────────────────────────────────

def add_features(df):
    """
    Compute technical indicators and lagged features.
    All features use only past data (no lookahead).
    """
    g = df.copy().sort_index()

    # Price-based indicators
    g['MA5']    = g['Close'].rolling(5).mean()
    g['MA20']   = g['Close'].rolling(20).mean()
    g['MA5_20'] = g['MA5'] / g['MA20']          # ratio (trend signal)

    # RSI (14-day)
    delta = g['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    g['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12-26 EMA difference)
    ema12       = g['Close'].ewm(span=12).mean()
    ema26       = g['Close'].ewm(span=26).mean()
    g['MACD']   = ema12 - ema26
    g['Signal'] = g['MACD'].ewm(span=9).mean()
    g['MACD_diff'] = g['MACD'] - g['Signal']

    # Volume change
    g['Vol_change'] = g['Volume'].pct_change()

    # Lagged log-returns (motivated by professor's time-series feedback)
    for lag in [1, 3, 5]:
        g[f'Return_{lag}d'] = np.log(g['Close'] / g['Close'].shift(lag))

    # Rolling volatility (10-day std of daily log-returns)
    daily_ret      = np.log(g['Close'] / g['Close'].shift(1))
    g['Volatility'] = daily_ret.rolling(10).std()

    # Target: 1 if tomorrow's close > today's close
    g['Target'] = (g['Close'].shift(-1) > g['Close']).astype(int)

    # Drop rows with NaN (from rolling windows)
    g.dropna(inplace=True)
    return g


FEATURE_COLS = [
    'MA5_20', 'RSI', 'MACD', 'MACD_diff',
    'Vol_change', 'Return_1d', 'Return_3d', 'Return_5d', 'Volatility'
]


def build_dataset(raw_data):
    """Apply feature engineering to all tickers and combine."""
    frames = []
    for ticker in TICKERS:
        sub = raw_data[raw_data['Ticker'] == ticker].copy()
        sub = add_features(sub)
        frames.append(sub)
    data = pd.concat(frames).sort_index()
    return data


# ─── 3. Train / Test Split ────────────────────────────────────────────────────

def split_data(data):
    train = data[data.index.year <  TEST_YEAR]
    test  = data[data.index.year == TEST_YEAR]
    X_train = train[FEATURE_COLS].values
    y_train = train['Target'].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test['Target'].values
    return X_train, y_train, X_test, y_test, train, test


def scale(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# ─── 4. LSTM Sequence Builder ─────────────────────────────────────────────────

def make_sequences(X, y, seq_len=SEQ_LEN):
    """Convert flat feature matrix to overlapping sequences for LSTM."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ─── 5. Model Training ────────────────────────────────────────────────────────

def train_logistic(X_tr, y_tr):
    cw = dict(zip([0, 1], compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)))
    m  = LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_SEED)
    m.fit(X_tr, y_tr)
    return m


def train_random_forest(X_tr, y_tr):
    cw = dict(zip([0, 1], compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)))
    m  = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight=cw,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m


def train_xgboost(X_tr, y_tr):
    ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           scale_pos_weight=ratio, eval_metric='logloss',
                           random_state=RANDOM_SEED, verbosity=0)
    m.fit(X_tr, y_tr)
    return m


def train_lstm(X_tr, y_tr, X_te, y_te):
    """Build and train a 2-layer LSTM with early stopping."""
    Xs_tr, ys_tr = make_sequences(X_tr, y_tr)
    Xs_te, ys_te = make_sequences(X_te, y_te)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_tr.shape[1])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(
        Xs_tr, ys_tr,
        validation_data=(Xs_te, ys_te),
        epochs=60, batch_size=64,
        callbacks=[es], verbose=0
    )
    return model, history, Xs_te, ys_te


# ─── 6. Evaluation ────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    return {
        'Model'    : name,
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall'   : recall_score(y_true, y_pred, zero_division=0),
        'F1'       : f1_score(y_true, y_pred, zero_division=0),
    }


def baseline_pred(y_true):
    """Always predict 'up' (majority class)."""
    return np.ones(len(y_true), dtype=int)


# ─── 7. Paper Trading Simulation ─────────────────────────────────────────────

def paper_trade(test_df, predictions, model_name):
    """
    Simple long-only strategy:
      Buy at today's close, sell at tomorrow's close when model predicts UP.
    Returns daily strategy returns and buy-and-hold returns.
    """
    # Align predictions with test_df (LSTM loses SEQ_LEN rows)
    df = test_df.copy().iloc[-len(predictions):]
    df['Pred']       = predictions
    df['Daily_ret']  = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    df['Strat_ret']  = df['Pred'].shift(1).fillna(0) * df['Daily_ret']
    df['Cum_strat']  = df['Strat_ret'].cumsum().apply(np.exp)
    df['Cum_bh']     = df['Daily_ret'].cumsum().apply(np.exp)
    sharpe = (df['Strat_ret'].mean() / (df['Strat_ret'].std() + 1e-9)) * np.sqrt(252)
    return df[['Cum_strat', 'Cum_bh']], sharpe


# ─── 8. Plotting ──────────────────────────────────────────────────────────────

def plot_accuracy_comparison(results_df):
    """Bar chart comparing accuracy and F1 of all models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = ['#AAAAAA', '#3B82F6', '#10B981', '#F4A428', '#8B5CF6']

    for ax, metric in zip(axes, ['Accuracy', 'F1']):
        vals = results_df[metric].values * 100
        bars = ax.bar(results_df['Model'], vals, color=palette, width=0.55,
                      edgecolor='white', linewidth=0.8)
        ax.axhline(results_df.loc[results_df['Model'] == 'Baseline', metric].values[0] * 100,
                   color='red', linestyle='--', linewidth=1.2, label='Baseline')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(45, 75)
        ax.set_ylabel(f'{metric} (%)', fontsize=11)
        ax.set_title(f'Model Comparison — {metric}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


def plot_confusion_matrices(cms, model_names):
    """2×2 grid of normalised confusion matrices."""
    n = len(model_names)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4.2))
    axes = axes.flatten()

    for ax, cm, name in zip(axes, cms, model_names):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                    linewidths=0.5, linecolor='white', cbar=False,
                    annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle('Confusion Matrices — Normalised (Test Year 2023)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


def plot_feature_importance(rf_model, xgb_model):
    """Side-by-side feature importance for RF and XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#3B82F6', '#F4A428']

    for ax, model, name, color in zip(axes,
                                       [rf_model, xgb_model],
                                       ['Random Forest', 'XGBoost'],
                                       colors):
        imp  = model.feature_importances_
        idx  = np.argsort(imp)
        ax.barh([FEATURE_COLS[i] for i in idx], imp[idx],
                color=color, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{name} — Feature Importance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


def plot_lstm_training(history):
    """Training and validation loss/accuracy curves for LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, label in zip(axes,
                                  ['loss', 'accuracy'],
                                  ['Loss (Binary Cross-Entropy)', 'Accuracy']):
        ax.plot(history.history[metric],          label='Train', color='#3B82F6', linewidth=2)
        ax.plot(history.history[f'val_{metric}'], label='Validation', color='#F4A428',
                linewidth=2, linestyle='--')
        ax.set_title(f'LSTM Training — {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_lstm_training.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


def plot_cumulative_returns(all_trade_results):
    """Cumulative return curves for all strategies vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors  = ['#3B82F6', '#10B981', '#F4A428', '#8B5CF6']
    bh_plotted = False

    for (name, df_trade, sharpe), color in zip(all_trade_results, colors):
        ax.plot(df_trade.index, df_trade['Cum_strat'], label=f'{name} (Sharpe={sharpe:.2f})',
                color=color, linewidth=1.8)
        if not bh_plotted:
            ax.plot(df_trade.index, df_trade['Cum_bh'], label='Buy & Hold',
                    color='#333333', linewidth=1.5, linestyle='--')
            bh_plotted = True

    ax.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_title('Paper Trading — Cumulative Returns (Test Year 2023)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Growth of $1 Invested', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_cumulative_returns.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


# ─── 9. Main Pipeline ─────────────────────────────────────────────────────────

def main(skip_lstm=False):
    print("=" * 60)
    print("  Stock Price Direction Prediction — ML Pipeline")
    print("=" * 60)

    # 1. Load data
    raw = load_yfinance(TICKERS, START_DATE, END_DATE)
    if raw is None:
        raw = generate_synthetic_data(TICKERS, START_DATE, END_DATE)

    # 2. Feature engineering
    data = build_dataset(raw)
    print(f"[INFO] Feature matrix: {data.shape[0]} samples, {len(FEATURE_COLS)} features.")

    # 3. Split and scale
    X_tr, y_tr, X_te, y_te, train_df, test_df = split_data(data)
    X_tr_s, X_te_s, scaler = scale(X_tr, X_te)
    print(f"[INFO] Train: {len(X_tr)} | Test: {len(X_te)} (year {TEST_YEAR})")

    # 4. Train models
    print("\n[TRAINING] Logistic Regression ...")
    lr  = train_logistic(X_tr_s, y_tr)

    print("[TRAINING] Random Forest ...")
    rf  = train_random_forest(X_tr_s, y_tr)

    print("[TRAINING] XGBoost ...")
    xgb_m = train_xgboost(X_tr_s, y_tr)

    lstm_model, lstm_history, Xs_te_lstm, ys_te_lstm = None, None, None, None
    if not skip_lstm:
        print("[TRAINING] LSTM (this may take a few minutes) ...")
        lstm_model, lstm_history, Xs_te_lstm, ys_te_lstm = train_lstm(X_tr_s, y_tr, X_te_s, y_te)

    # 5. Predict
    y_pred_bl   = baseline_pred(y_te)
    y_pred_lr   = lr.predict(X_te_s)
    y_pred_rf   = rf.predict(X_te_s)
    y_pred_xgb  = xgb_m.predict(X_te_s)
    y_pred_lstm = None
    if lstm_model is not None:
        y_pred_lstm = (lstm_model.predict(Xs_te_lstm, verbose=0).flatten() > 0.5).astype(int)

    # 6. Evaluate
    results = [
        evaluate('Baseline',         y_te, y_pred_bl),
        evaluate('Logistic Reg.',    y_te, y_pred_lr),
        evaluate('Random Forest',    y_te, y_pred_rf),
        evaluate('XGBoost',          y_te, y_pred_xgb),
    ]
    if y_pred_lstm is not None:
        results.append(evaluate('LSTM', ys_te_lstm, y_pred_lstm))

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("  RESULTS TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    results_df.to_csv('results_summary.csv', index=False)
    print("\n[SAVED] results_summary.csv")

    # 7. Confusion matrices
    cms   = [confusion_matrix(y_te, p) for p in [y_pred_bl, y_pred_lr, y_pred_rf, y_pred_xgb]]
    names = ['Baseline', 'Logistic Regression', 'Random Forest', 'XGBoost']
    if y_pred_lstm is not None:
        cms.append(confusion_matrix(ys_te_lstm, y_pred_lstm))
        names.append('LSTM')

    # 8. Paper trading
    trade_results = []
    for name, preds, y_true_len, df_sub in [
        ('Logistic Reg.',  y_pred_lr,   len(y_te),           test_df),
        ('Random Forest',  y_pred_rf,   len(y_te),           test_df),
        ('XGBoost',        y_pred_xgb,  len(y_te),           test_df),
    ]:
        # Use AAPL for the trade simulation (representative single stock)
        apple_test = test_df[test_df['Ticker'] == 'AAPL'].copy()
        apple_preds = preds[:len(apple_test)]   # align to single stock
        df_tr, sharpe = paper_trade(apple_test, apple_preds, name)
        trade_results.append((name, df_tr, sharpe))
        print(f"[TRADE] {name:20s} | Sharpe = {sharpe:.3f}")

    if y_pred_lstm is not None:
        apple_test = test_df[test_df['Ticker'] == 'AAPL'].copy()
        apple_preds_lstm = y_pred_lstm[:len(apple_test)]
        df_tr, sharpe = paper_trade(apple_test, apple_preds_lstm, 'LSTM')
        trade_results.append(('LSTM', df_tr, sharpe))
        print(f"[TRADE] {'LSTM':20s} | Sharpe = {sharpe:.3f}")

    # 9. Plots
    print("\n[PLOTTING] Generating figures ...")
    plot_accuracy_comparison(results_df)
    plot_confusion_matrices(cms, names)
    plot_feature_importance(rf, xgb_m)
    if lstm_history is not None:
        plot_lstm_training(lstm_history)
    plot_cumulative_returns(trade_results)

    print("\n[DONE] All figures saved to:", FIGURES_DIR)
    print("[DONE] Pipeline complete.")
    return results_df, trade_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Price Direction Prediction')
    parser.add_argument('--no-lstm', action='store_true',
                        help='Skip LSTM training (faster testing)')
    args = parser.parse_args()
    main(skip_lstm=args.no_lstm)
