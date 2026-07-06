# DeepEdgeBenchmark — run report

Generated: 2026-07-06 12:05:54

## Run configuration

- Ticker: `SPY`
- Window: `2023-01-01` -> `2024-12-31`
- Test ratio: `0.15`
- Seed: `42`

## Library versions

- numpy: `2.2.6`
- pandas: `3.0.3`
- statsmodels: `0.14.6`
- sklearn: `1.9.0`
- tensorflow: `2.21.0`
- prophet: `1.3.0`
- arch: `8.0.0`
- yfinance: `1.5.1`

## Test suite (models/)

- Result: **PASS**
- Summary: `38 passed, 36 warnings in 7.84s`
- Duration: 8.8s

## Metrics (out-of-sample, walk-forward)

| model | ticker | start | end | test_ratio | n_test | RMSE | MAE | MAPE | DirAcc_% | PI_cov95_% | runtime_s | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ARIMA | SPY | 2023-01-01 | 2024-12-31 | 0.15 | 76 | 4.345 | 3.1224 | 0.55 | 46.67 | 96.05 | 0.24 | 42 |
| SARIMA | SPY | 2023-01-01 | 2024-12-31 | 0.15 | 76 | 4.3328 | 3.1175 | 0.54 | 49.33 | 94.74 | 3.78 | 42 |
| Prophet | SPY | 2023-01-01 | 2024-12-31 | 0.15 | 76 | 42.1249 | 40.4472 | 7.04 | 53.33 | 3.95 | 0.11 | 42 |
| LSTM | SPY | 2023-01-01 | 2024-12-31 | 0.15 | 76 | 10.6015 | 9.373 | 1.62 | 54.67 | 94.74 | 3.29 | 42 |
