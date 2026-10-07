# KrishiSetu crop recommender (IMP project)

Forecasts a district's weather for the next 8 months, validates every crop against rule-based water and temperature requirements, and recommends the best crop and sowing month, with a probability that it stays feasible.

**Start with [`TEAM_GUIDE.md`](TEAM_GUIDE.md)**: it explains every model, the full flow, the outputs, how to extend the code, and what is and is not verified.

## Quick start

```bash
pip install kagglehub pandas numpy scikit-learn

python tests/test_smoke.py                                  # fast sanity check on fake data
python crop_reco.py                                         # baseline
python crop_reco.py --state maharashtra --district pune --start 2026-10
python crop_reco.py --weather-model ar_trend --scorer ml_yield
python crop_reco.py --compare                               # all weather models x all scorers
python crop_reco.py --list-models
```

## Models at a glance

| Slot | Name | Idea |
|---|---|---|
| Weather | `linear_trend` | Baseline: straight-line trend per season (deterministic) |
| Weather | `ar_trend` | Trend + AR(1) year-to-year variability, many sampled futures |
| Weather | `bootstrap` | Trend + resampled real historical years, many sampled futures |
| Scorer | `rule_based` | Baseline: water rule, temperature, humidity, wind, local history |
| Scorer | `ml_yield` | Rules + gradient-boosting yield model trained on the dataset |

New models are plug-ins: one small class plus a decorator (see TEAM_GUIDE section 9).

## Things to check on your first real run

1. The printed weather summary and the **precipitation unit guess** (override with `--precip-unit`).
2. The **season-to-month mapping** in `krishi/config.py`.
3. For `ml_yield`: the held-out MAE must beat the baseline MAE, or don't trust it.
4. The ideal crop conditions are **synthetic placeholders**.

## Files

- `crop_reco.py`: entry point
- `krishi/`: the package (weather models, scorers, pipeline, CLI)
- `legacy/crop_reco_v1.py`: the original single-script baseline
- `tests/`: smoke tests and a synthetic dataset
- `output/`: generated CSVs
- `CODE_WALKTHROUGH.md`: detailed notes on the original single script (`legacy/`)
