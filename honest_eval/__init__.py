"""
honest_eval — an honest, discriminating evaluation layer for DeepEdgeBenchmark
==============================================================================
Implements IMPROVEMENTS_BRIEF.md:

  * Point 0  naive.py      — correct random-walk baseline + acceptance verifier
  * Point 1  metrics.py    — score *changes* not levels (MASE, Theil's U,
                             change-correlation, directional accuracy + binomial
                             CI, Diebold-Mariano vs the corrected naive)
  * Point 2  validation.py — robust walk-forward (expanding vs fixed rolling)
                             and purged/embargoed blocked CV (Lopez de Prado)
  * Point 3  multistep.py  — dense daily rolling-origin multi-horizon eval
                             (D+1 / D+7 / D+30) with Newey-West DM and
                             error-vs-horizon degradation curves
  * Point 4  targets.py    — reformulated targets where signal may exist
                             (volatility: QLIKE/PIT/Winkler; direction: AUC/Brier)

Every module is pure numpy/scipy/sklearn (no network, no matplotlib) so the
statistics can be unit-tested offline.  The heavy model-dependent parts take a
generic ``forecast_path`` callable so they work with the repo's ARIMA/SARIMA/
Prophet/LSTM runners *or* a cheap analytic forecaster in tests.
"""

from . import metrics, naive, validation, multistep, targets

__all__ = ["metrics", "naive", "validation", "multistep", "targets"]
