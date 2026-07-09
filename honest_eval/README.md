# honest_eval — implementation of IMPROVEMENTS_BRIEF.md

An honest, discriminating evaluation layer for DeepEdgeBenchmark. The dashboard
analysis showed all models were effectively predicting *"yesterday's price ± ε"*:
`corr(pred_t, price_{t-1}) ≈ 1.000`, change-correlation ≈ 0, directional accuracy
43–54 % — an **evaluation artifact** of a re-anchoring 1-step walk-forward scored
on *levels*. This package makes the benchmark measure what matters.

## What maps to which point of the brief

| Brief | Module | Delivers |
|---|---|---|
| **Point 0** — fix naive baseline | [`naive.py`](naive.py) | Single source of truth for the random-walk baseline (no injected noise) + `verify_naive()` acceptance check: predictions must equal the previous close *exactly* and `|RMSE_dash − RMSE_ref|/RMSE < 0.1%`. |
| **Point 1** — score changes not levels | [`metrics.py`](metrics.py) | `theil_u`, `mase`, `change_correlation`, `directional_accuracy` (Wilson 95 % CI + binomial p vs coin), `diebold_mariano` (Newey-West HAC + HLN correction), `skill_verdict`. |
| **Point 2** — robust validation | [`validation.py`](validation.py) | `walk_forward_splits` (expanding vs fixed rolling), `compare_windows`, `purged_kfold_splits` (purge + embargo, López de Prado), `subperiod_report`, `volatility_regimes`. |
| **Point 3** — multi-step without re-anchoring | [`multistep.py`](multistep.py) | Dense **daily** rolling origin for D+1/7/30 (~150+ evals), per-horizon scoring, `error_vs_horizon` degradation curves, Newey-West DM (lag ≥ h−1). Model adapters refit at each origin. |
| **Point 4** — reformulated targets | [`targets.py`](targets.py) | Volatility (`evaluate_volatility`: QLIKE, MSE-var, PIT, Winkler vs persistence/EWMA/GARCH), direction (`evaluate_direction`: AUC, Brier, binomial vs 50 %, always-up/majority baselines), returns+exogenous features (`evaluate_returns_with_features`, purged-CV tuned ridge). |
| dashboard | [`report.py`](report.py) | Variations panel (Δpred vs Δreal scatter + change series, levels demoted), honest KPI table, error-vs-horizon curves, vol/direction tabs. |

Driver: [`../honest_benchmark.py`](../honest_benchmark.py) wires the library to the
repo's ARIMA/SARIMA/Prophet/LSTM runners.

## Run it

```bash
# offline synthetic smoke test (no network, no heavy training)
python honest_benchmark.py --demo --models arima --out honest.png

# real data — heavy models refit per origin, so thin origins with --step
python honest_benchmark.py --ticker BTC-USD --models arima sarima --step 3 --curve-max 30
python honest_benchmark.py --offline prices.csv --models arima --no-multistep

# the maths, unit-tested offline (39 checks, no network)
python -m honest_eval.tests.test_honest_eval
```

## Reading rule (Point 1)

**U ≈ 1 and DM not significant ⇒ the model adds nothing over the naive.** The KPI
table states this verdict in plain language per (model, asset, horizon). On the
current data expect `U ≈ 1.00` for ARIMA/SARIMA and `> 1` for LSTM/Prophet.

## Note on Point 0 scope

**Resolved.** The noise injection was traced to `models/naive_model.py` (a
uniform ±5% "drift by design", std 5%/√3 ≈ 2.9%) once the external `Run`
pipeline was merged into this repository. That file is now a **strict
persistence baseline** (pred = previous close exactly, Gaussian random-walk PI),
`benchmarks/multi_horizon.py`'s naive adapter follows the same convention
(point = last price, PI ∝ σ√h), and `models/test_naive_model.py` cross-checks
the output against `verify_naive()`. The skill metrics of this package
(Theil's U, MASE, DM, DirAcc±CI) are wired into `model_artifacts/pipeline.py`
(metrics payload) and surfaced by `model_artifacts/generate_dashboard.py`
(KPI cards + comparison table).
