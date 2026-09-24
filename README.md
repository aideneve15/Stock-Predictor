# Stock-Predictor

**Predicting next-day S&P 500 direction with a walk-forward-validated Random Forest**

## Overview

This project builds a binary classifier that predicts whether the S&P 500 (`^GSPC`) will close **higher** the next trading day, using historical price and volume data. The notebook walks through the full process: pulling data, building a naive baseline model, showing why a simple train/test split is misleading for time series, and then improving the model with engineered features, an auxiliary international-market signal, and hyperparameter tuning — all validated with a proper walk-forward backtest.

## Data

- **Target asset**: S&P 500 index (`^GSPC`), full history via `yfinance`.
- **Auxiliary asset**: `EFA` (iShares MSCI EAFE ETF, exposure to developed markets outside the US/Canada), added later as a feature to test whether international market moves help predict the S&P.
- **Target variable**: `1` if tomorrow's close is higher than today's close, else `0`.
- Modeling is restricted to 1990–present.

## Approach

1. **Baseline model** — `RandomForestClassifier` on raw `Open/High/Low/Close/Volume`, trained on all data except the last 100 days and tested on those 100 days.
2. **Why that's misleading** — a single 100-day holdout gives an overly optimistic read. The notebook replaces it with a **walk-forward backtest**: train on the first 10 years, test on the following year, then roll forward one year at a time, so every prediction is made on genuinely unseen future data.
3. **Feature engineering** — rolling-average ratios and up/down "trend" counts over five horizons (2, 5, 60, 250, and 1000 trading days), giving the model short- and long-term context instead of raw prices alone.
4. **Probability thresholding** — predictions use `predict_proba` with a 0.6 cutoff instead of the default 0.5, trading recall for higher-precision "up" signals.
5. **Auxiliary features** — rolling ratios from `EFA` are added as additional predictors.
6. **Hyperparameter tuning** — `GridSearchCV` with `TimeSeriesSplit` (5 folds) tunes `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`, optimizing for precision.

## Results

- Naive baseline ("always predict up"): the S&P 500 has historically closed higher on roughly 55% of days — the bar any real model needs to clear.
- First Random Forest model: **0.531 precision**.
- After rolling-average features, the EFA signal, and tuning: **~0.55 precision** in the backtest.

The notebook is candid about the outcome: after feature engineering and tuning, the model performs roughly in line with the "market goes up most days" baseline rather than decisively beating it — a realistic result for next-day direction prediction on a single index.

## Requirements

```
pandas
matplotlib
yfinance
scikit-learn
```

## Usage

Open `SPY.ipynb` and run the cells top to bottom. An internet connection is required since data is pulled live via `yfinance` on each run.

## Disclaimer

This is an educational project exploring time-series modeling and backtesting methodology. It is **not** financial advice, and the model's precision is close to a naive baseline — it should not be used to make real trading decisions.
