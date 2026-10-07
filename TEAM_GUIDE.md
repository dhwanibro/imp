# KrishiSetu crop recommender: team guide

Everything the team needs to understand the models, the data flow, the outputs, and how to extend or challenge the code. Written for teammates with different backgrounds: Sections 1 to 3 need no ML knowledge; Sections 5 to 7 go into the models; Section 10 is the honest status of what is and is not verified.

Contents
1. What this is
2. Quick start
3. The flow, end to end
4. The data
5. Weather models (3)
6. Crop scorers (2) and the fake crop conditions
7. Turning many forecasts into one recommendation
8. Outputs and column dictionary
9. How to extend it
10. Status: what is verified, what is not
11. Which combination should we use?
12. FAQ and glossary
13. Repo map

---

## 1. What this is

The program answers: **"For this district, which crop should be sown in which month over the next 8 months?"**

It does that in three moves:

1. **Forecast the weather** for the district, month by month, for the next 8 months (temperature, rainfall, humidity, wind).
2. **Check every crop against that weather.** For each crop and each possible sowing month, ask: is there enough water (rule-based check)? Is the temperature suitable? Has this crop actually been grown here in this season? How well has it yielded?
3. **Rank and recommend.** Pick the best feasible crop and sowing month, and say how confident the weather forecast is that it stays feasible.

It is a **modular baseline**. Weather models and crop scorers are plug-ins: you choose one of each at run time, or compare all combinations.

| Slot | Options | One-line idea |
|---|---|---|
| Weather model | `linear_trend` | Baseline. A straight line through each season's history. |
| | `ar_trend` | The line plus realistic year-to-year wobble that fades with distance. Gives many possible futures. |
| | `bootstrap` | The line plus the wobble of a randomly chosen real past year. Gives many possible futures that keep real correlations. |
| Crop scorer | `rule_based` | Baseline. Water rule, temperature, humidity, wind, local history. |
| | `ml_yield` | Same rules, plus a trained model that predicts how well the crop should yield under the forecast weather. |

---

## 2. Quick start

```bash
pip install kagglehub pandas numpy scikit-learn

python tests/test_smoke.py                   # sanity check, uses fake data, takes seconds

python crop_reco.py                          # baseline on the busiest district in the data
python crop_reco.py --state maharashtra --district pune --start 2026-10
python crop_reco.py --weather-model ar_trend --scorer ml_yield
python crop_reco.py --compare                # every weather model x every scorer
python crop_reco.py --list-models
python crop_reco.py --csv path/to/file.csv   # skip the Kaggle download
```

| Flag | Default | Meaning |
|---|---|---|
| `--state`, `--district` | busiest district | Region to model (case-insensitive). |
| `--start` | next month | First forecast month, `YYYY-MM`. |
| `--horizon` | 8 | Months to forecast. |
| `--irrigation` | 350 | Irrigation water assumed available per crop season, in mm. A placeholder; see FAQ. |
| `--precip-unit` | auto | `mm_day`, `mm_month` or `mm_season`. Overrides the unit guess. |
| `--weather-model` | `linear_trend` | See Section 5. |
| `--scorer` | `rule_based` | See Section 6. |
| `--samples` | 100 | How many possible futures to draw (probabilistic models only). |
| `--min-prob` | 0.5 | A crop must pass the feasibility rules in at least this share of the futures to be recommended. |
| `--seed` | 0 | Makes the sampling reproducible. |
| `--compare` | off | Run all combinations and print a comparison table. |

Python files are run from the `imp/` folder. `python -m krishi ...` also works.

---

## 3. The flow, end to end

```
                      Kaggle CSV (district-level, India)
                                   |
                         data.standardise()         clean names, types, units
                                   |
              +--------------------+---------------------+
              |                                          |
   weather history by                           crop history: area, production,
   (district, year, season)                     which crops grown where and when
   (state data as fallback)                              |
              |                                          |
      WEATHER MODEL  (choose 1)                          |
      linear_trend | ar_trend | bootstrap                 |
              |                                          |
   monthly forecast, next 8 months                       |
   (1 table, or 100 sampled tables)                      |
              |                                          |
              +-----------------> CROP SCORER (choose 1) <--- fake ideal crop conditions
                                  rule_based | ml_yield       (18 crops, synthetic)
                                           |
                       one scored table per forecast sample:
                       every (crop x sowing month), with feasible yes/no + score
                                           |
                                  aggregate.py
                       p_feasible, expected score, bad-year score
                                           |
                            best crop + sowing month, CSVs in output/
```

Step by step, in the order `cli.py` runs them:

1. **Load and clean** (`data.py`). Find the columns, normalise text, convert units if needed (Section 4).
2. **Pick the region.** The district you asked for, or the one with most rows.
3. **Build the weather history** for that district: one weather value per (year, season), averaged across crops. The state's history is built too, as a fallback.
4. **Fit the weather model** on that history.
5. **Backtest it.** Refit without the last 3 years, predict them, report the error (Section 5.4).
6. **Produce forecasts.** The central forecast is a table of 8 months. Probabilistic models also produce 100 sampled tables.
7. **Fit the crop scorer.** The rule-based scorer learns the yield priors and where each crop is grown. The ML scorer additionally trains a model.
8. **Score each forecast table.** Each table yields a ranking of (crop, sowing month) options.
9. **Aggregate** across tables into probabilities and an expected score.
10. **Print and save** the recommendation and CSVs.

### Worked example (made-up mock data, not real results)

Baseline run, Pune, forecast from October 2026:

| Month | Temp (C) | Rain (mm) | Humidity | Wind |
|---|---|---|---|---|
| Oct | 27.1 | 81.7 | 63.5 | 2.05 |
| Nov | 24.0 | 46.5 | 72.0 | 1.97 |
| Dec | 24.0 | 46.5 | 72.0 | 1.97 |

Groundnut sown in October (a 4-month crop) sees about 221 mm of rain in its window. Its water need is about 513 mm. With 350 mm of irrigation: supply = 0.8 x 221 + 350 = 527 mm, ratio = 1.03, so the water rule passes. Temperature is inside its ideal range, it is grown in this district, so it ranks first. Notice Nov and Dec are identical: that is a property of the data (Section 12).

---

## 4. The data

Source: `vihith12/crop-yield-prediction-dataset` (via `kagglehub`). One row per district, year, season and crop. Columns used:

| Dataset column | Used for |
|---|---|
| `state_names`, `district_names` | Picking the region. |
| `crop_year`, `season_names` | Time axis of the weather history; matching crops to seasons. |
| `temperature`, `precipitation`, `humidity`, `wind_speed` | The four weather variables that are forecast. |
| `crop_names`, `area`, `production` | Yield history (yield = production / area), where each crop is grown. |
| `soil_type`, `N`, `P`, `K` | Features of the ML yield model only. |

Two things the data does **not** have, which shaped the design:

- **No months.** Only seasons (Kharif, Rabi, Summer, ...). The code maps seasons to months with `SEASON_MONTHS` in `krishi/config.py` (for example Kharif = Jun to Oct, Rabi = Nov to Mar). Each forecast month is the **average of the seasons that cover it**. `SEASON_SOW` says when each season's crops are normally sown. **Both are our assumptions; change them in `config.py` if the team knows better.**
- **No documented precipitation unit.** The code guesses: if the median positive value is below 25 it treats the column as **mm per day**, otherwise as **mm per season**. It prints the guess on every run. Rainfall drives the water rule, so a wrong guess changes recommendations. Check the printed min/median/max, and override with `--precip-unit` if needed.

Cleaning also: strips whitespace and lowercases text (so `"Kharif   "` becomes `kharif`), converts Kelvin to Celsius if temperatures are above 100, and drops rows missing year, temperature or rainfall.

---

## 5. Weather models

All three share the same skeleton (`krishi/weather/forecaster.py`):

1. For each **season** and each **variable**, take the last 15 years of history (district first, state if the district has fewer than 4 years of that season).
2. Fit a straight-line trend: `value = slope x year + intercept`.
3. Predict the value for the target year, and turn seasonal values into monthly ones (Section 4).
4. Rainfall is converted to **mm per month** using the unit guess.

What differs is **what happens around the trend line**.

### 5.1 `linear_trend` (baseline)

Just the line, clipped to the smallest and largest values ever observed, so a trend can never produce impossible numbers. Deterministic: one forecast.

- **Good:** simple, explainable, fast.
- **Bad:** claims a single certain future. Cannot express "a dry year is possible". If the data ends years before the forecast, it is a pure extrapolation.

### 5.2 `ar_trend` (probabilistic)

Idea: real weather does not sit on a straight line. Each year deviates, and a deviation tends to carry a little into the next year (a wet year is slightly more likely to follow a wet year). That is an **AR(1) process** on the deviations ("residuals") from the line.

How it works:

- **Residual** = actual value minus trend for each past year.
- **phi** = how strongly one year's residual carries into the next, estimated from consecutive-year pairs (needs 5 pairs, otherwise 0; clipped to between -0.5 and 0.9).
- **Central forecast** = trend + phi^gap x (last observed residual). `gap` is how many years the target is after the last data year. The carry-over fades fast, so for distant years the central forecast is just the trend.
- **Possible futures:** start from the last residual and simulate forward year by year with random noise of the right size. For a long gap, the spread settles at the historical variability.
- Values are clipped to the observed range plus 10% of its span, and to physical limits (rain and wind cannot be negative, humidity stays within 0 to 100).

Each variable is simulated **independently**, which is its main weakness.

### 5.3 `bootstrap` (probabilistic, non-parametric)

Idea: instead of assuming a mathematical shape for the wobble, **reuse real history**.

- Compute the residual vector for every historical year: one number for every (season, variable).
- For each possible future, pick a **random past year** and add *that whole year's* residuals to the trend. Every season and every variable gets the same past year.
- Because whole years are reused, real relationships survive: a dry Kharif tends to come with a warm Kharif, humidity moves with rain, and so on. The AR model loses this.

The central forecast equals the trend. If a (season, variable) series has a missing year, its residual is filled with 0 for that year.

### 5.4 How to compare weather models: the backtest

Every run prints a backtest: the model is refit **without the last 3 years**, asked to predict them, and scored against what actually happened. It needs at least 9 years of history.

| Metric | Meaning | Good |
|---|---|---|
| `temp_MAE` | Average error of seasonal temperature, in C | Lower |
| `precip_MAE` | Average error of seasonal precipitation, in dataset units | Lower |
| `temp_cov80`, `precip_cov80` | (Probabilistic models) share of true values that fell inside the models' 10th to 90th percentile band | Near 0.80 |

A model that is *too confident* shows cov80 well below 0.8. A model that is too vague shows it near 1.0. The three models have similar MAE because they share the same trend line; the probabilistic ones differ in whether their uncertainty is honest. Note that the backtest holds out only 3 years x a few seasons, so cov80 is a **noisy** estimate; do not over-read a single run.

### 5.5 Comparison

| | `linear_trend` | `ar_trend` | `bootstrap` |
|---|---|---|---|
| Output | 1 forecast | 100 futures | 100 futures |
| Expresses uncertainty | No | Yes | Yes |
| Keeps relationships between variables | n/a | No | Yes |
| Distributional assumption | n/a | Normal noise | None |
| Needs history | 4+ years/season | 4+ years (5+ pairs for phi) | 4+ years |
| Weakness | Overconfident | Variables independent | Can only repeat what was seen; short history = few distinct futures |

None of them is a real weather forecast (no atmospheric information). They are **trend + historical variability** models. Their honest role is "what is the typical climate for those months, and how variable is it", not "will it rain on a given date".

---

## 6. Crop scorers

A scorer receives one monthly forecast table and returns, for every crop and every sowing month that fits inside the forecast, a row with `feasible` (yes/no) and `score` (higher is better).

### 6.1 The ideal crop conditions are FAKE

`krishi/crops.py` holds a synthetic table for 18 Indian crops: Rice, Wheat, Maize, Bajra, Jowar, Ragi, Gram, Tur, Moong, Urad, Groundnut, Soyabean, Sunflower, Cotton, Mustard, Potato, Onion, Sesamum. For each crop it defines:

| Field | Meaning |
|---|---|
| `season_months` | Length of the growing window (3 to 6 months) |
| `water_need_mm` | Water needed over the whole season |
| `temp_opt_min/max` | Temperature range where the crop does best |
| `temp_abs_min/max` | Outside this range the crop fails |
| `hum_opt_min/max` | Comfortable humidity range |
| `wind_max` | Wind speed above which growth suffers |

The values are plausible textbook-style numbers with a small seeded random jitter, saved to `output/crop_conditions.csv`. **They are placeholders. Replacing them with sourced agronomy data (ICAR, TNAU, FAO EcoCrop) is the most valuable single improvement to recommendation quality.** The scorers do not care where the table comes from, as long as the columns are the same. Sugarcane and other crops longer than the horizon are excluded.

### 6.2 `rule_based` (baseline)

For each crop and each sowing month, over the growth window:

1. **Water check (hard rule).**
   `supply = 0.8 x (forecast rain in window) + irrigation`; `ratio = supply / water_need`.
   Passes if `ratio >= 0.85`. The water score is `min(ratio, 1)`, with a mild penalty if `ratio > 1.5` (waterlogging).
2. **Temperature suitability (hard rule).** Score 1 inside the optimal range, falling to 0 at the absolute limits, averaged over the months of the window. Must be at least `0.4`.
3. **Humidity factor (soft).** `0.5 + 0.5 x humidity score`, so it can cut the score by at most half.
4. **Wind factor (soft).** 1 unless mean wind exceeds the crop's limit; then `max(0.5, limit / wind)`.
5. **Local-history factor (soft).** Was this crop recorded in the district in the season matching the sowing month? **1.0** yes; **0.85** only in the state; **0.5** neither.
6. **Yield prior (soft).** The district's historical yield for the crop compared with the national yield of the same crop, mapped to 0 to 1 (0.5 means "same as national average").

```
rule_core = temperature_score x water_score x humidity_factor x wind_factor x local_history_factor
score     = rule_core x (0.7 + 0.3 x yield_prior)
feasible  = water rule passed AND temperature_score >= 0.4
```

Only water and temperature can veto a crop; everything else only reduces its score. All the factors and thresholds are class constants at the top of `scoring/rule_based.py`.

### 6.3 `ml_yield` (hybrid)

The fake ideal-condition table knows nothing about how crops really performed. This scorer adds a model trained on the district-level history.

**What it learns.** Target: **relative yield** = (production / area) divided by that crop's national average yield in the dataset (clipped at the 99th percentile to ignore outliers). A value of 1.0 means "average for that crop". Dividing by the crop's own average makes potato and sesamum comparable.

**Features:** temperature, precipitation, humidity, wind speed (that season's weather), N, P, K, soil type, season, crop.

**Model:** scikit-learn `HistGradientBoostingRegressor` (gradient-boosted trees; handles missing values and categories). Trained on at most 200,000 rows.

**How it is used.** For each candidate (crop, sowing month), the forecast weather of its growing window (converted back into the dataset's precipitation unit), the district's most common soil type and average N, P, K go in; a predicted relative yield comes out. Then:

```
ml_score = clip(predicted_relative_yield / 2, 0, 1)        (0.5 = national average)
score    = rule_core x (0.4 + 0.6 x ml_score)
feasible = unchanged: the same rule-based water and temperature check
```

So the **rules still validate** feasibility, as the brief required; the ML model only re-ranks the feasible options.

**How to judge it (important).** At training time it prints:

```
ML yield model, held-out last 3 years: {holdout_r2, holdout_mae, baseline_mae}
```

It is trained on older years and tested on the last 3. `baseline_mae` is the error of "always predict the national average". **If `holdout_mae` is not clearly below `baseline_mae`, the model has no skill and its ranking should not be trusted.** (On our fake test data it does not beat the baseline, which is expected because the fake yields are mostly noise. The check must be repeated on the real data.)

Also remember it learns **associations**, not causes, and it is only as good as the weather columns in the data, which are season-level averages.

---

## 7. Turning many forecasts into one recommendation

A probabilistic weather model produces 100 possible futures. The scorer ranks options inside each one. `scoring/aggregate.py` combines them, per (crop, sowing month):

| Quantity | Definition |
|---|---|
| `p_feasible` | Share of futures in which the water and temperature rules pass |
| `exp_score` | Average of (score if feasible, else 0) across futures. **The ranking value.** |
| `score_p10` | 10th percentile of the same: a "bad year" outcome |

**Recommendation = highest `exp_score` among options with `p_feasible >= --min-prob`** (default 0.5). If nothing reaches the threshold, the best option overall is shown and clearly marked as below the threshold.

With a deterministic model (`linear_trend`) there is one future, so `p_feasible` is 0 or 1 and the result matches the plain baseline ranking.

### Worked example of why this matters (mock data)

With little irrigation (120 mm) and `ar_trend`:

| Crop, sow Oct 2026 | p_feasible | exp_score | grown here? |
|---|---|---|---|
| Moong | 0.56 | 0.404 | yes |
| Gram | 0.48 | 0.285 | yes |
| Mustard | 0.67 | 0.172 | no |

At `--min-prob 0.6`, Moong has the best expected score but is only feasible in 56% of futures, so it is **excluded**, and Mustard is recommended: less attractive on paper but a safer bet on water. Two lessons for the team:

1. This is the risk-aware behaviour we want: the recommendation changes when the weather is uncertain.
2. **A design wrinkle:** the threshold can pick a crop that has never been grown in that district (`grown_in = none`). It still carries the 0.5 local-history penalty, but it can win. A real product might want to forbid that or flag it clearly. This is a team decision.

---

## 8. Outputs and column dictionary

Printed to the terminal and saved to `output/`:

| File | Contents |
|---|---|
| `weather_forecast.csv` | Monthly `temp_c`, `rain_mm`, `humidity_pct`, `wind_speed`; for probabilistic models also `temp_c_p10/p90`, `rain_mm_p10/p90` |
| `crop_conditions.csv` | The fake ideal-condition table used |
| `crop_recommendations.csv` | All (crop, sowing month) options, aggregated, best first |
| `model_comparison.csv` | (`--compare` only) one row per weather model x scorer |

`crop_recommendations.csv` columns:

| Column | Meaning |
|---|---|
| `crop`, `sow`, `harvest` | Crop, first and last month of its growth window |
| `season_rain_mm` | Average forecast rainfall in the window |
| `water_need_mm` | The crop's (fake) water requirement |
| `p_water_ok` | Share of futures passing the water rule |
| `p_feasible` | Share passing water **and** temperature rules |
| `temp_score` | Average temperature suitability, 0 to 1 |
| `grown_in` | `district`, `state` or `none` |
| `yield_prior` | 0 to 1; historical (rule_based) or ML-predicted (ml_yield); 0.5 = national average |
| `exp_score` | Expected score, counting infeasible futures as 0 |
| `score_p10` | Bad-year score |

---

## 9. How to extend it

The architecture exists so that a new idea is one small file, not a rewrite.

### 9.1 Add a weather model

A model implements two methods, everything else is inherited: `fit(ctx)` and `predict_season(season, var, year)`. Probabilistic models also implement `simulate(years, n, rng)` and set `probabilistic = True`.

```python
# krishi/weather/my_model.py
import numpy as np
from ..registry import WEATHER_MODELS
from .forecaster import SeasonalForecaster, fit_trends

@WEATHER_MODELS.register("historical_mean")
class HistoricalMean(SeasonalForecaster):
    def fit(self, ctx):
        self.trends = fit_trends(ctx)          # per (season, variable) history
        self.keys = list(self.trends)          # REQUIRED: which series exist
        return self

    def predict_season(self, season, var, year):
        t = self.trends.get((season, var))
        return float(t.values.mean()) if t is not None else np.nan
```

Then add `my_model` to the import line in `krishi/weather/__init__.py`. It immediately appears in `--weather-model`, `--list-models` and `--compare`. `tests/test_smoke.py` contains a working example of this.

Ideas worth trying: use real monthly climatology or IMD seasonal outlooks; ENSO/IOD indicators as predictors of monsoon rainfall; a model pooled across neighbouring districts.

### 9.2 Add a crop scorer

Implement `fit(ctx)` and `evaluate(forecast)`; return a table with at least the columns `crop, sow, harvest, season_rain_mm, water_need_mm, water_ok, temp_score, grown_in, yield_prior, score, feasible` (the rule-based scorer returns more). Register with `@SCORERS.register("name")` and import it in `krishi/scoring/__init__.py`. To reuse the hard validation, build on `RuleBasedScorer` like `MLYieldScorer` does.

Ideas: a soil-suitability filter using `soil_type` and N, P, K; a crop-coefficient water balance instead of the flat 0.8 x rain rule; an economic stage ranking feasible crops by expected profit (MSP, cost of cultivation).

### 9.3 Replace the fake crop conditions

Write a function that returns a DataFrame with the same columns as `generate_crop_conditions()` (see Section 6.1) and call it in `pipeline.prepare()`. No other code changes.

### 9.4 Change the season mapping or defaults

`krishi/config.py` (season to month, sowing windows) and the class constants at the top of the scorers (thresholds and weights).

---

## 10. Status: what is verified, what is not

**Verified (on synthetic data with the same columns as the Kaggle file; Python 3.12, scikit-learn 1.8):**

- `tests/test_smoke.py` passes: all 3 weather models x 2 scorers run; forecasts have the right shape and physical ranges; backtests run; a new model can be plugged in.
- `linear_trend` reproduces the original single-script baseline (`legacy/crop_reco_v1.py`) to within rounding.
- The CLI runs in single-run and `--compare` modes, and probabilities behave sensibly when irrigation is scarce.

**Not verified:**

- **The real Kaggle dataset has not been run through the final package.** The earlier version of the loader found the real column names, but the full pipeline, the `ml_yield` training (size, time, scikit-learn version differences) and the precipitation-unit guess are untested on real data. Run `python crop_reco.py` and read the printed summary first.
- **Nothing validates that recommendations are agronomically right.** The backtest only checks weather. There is no comparison with what farmers or agronomists actually plant.
- **The ideal crop conditions are synthetic**, so every recommendation is only as meaningful as those placeholders.
- **`ml_yield` has no demonstrated skill yet** (see Section 6.3).
- **Data age.** If the dataset's last year is far before the forecast date, the weather models extrapolate across the gap. The probabilistic ones widen their uncertainty accordingly; the baseline does not.

---

## 11. Which combination should we use?

| Goal | Use |
|---|---|
| Explain the method to someone new / demo | `linear_trend` + `rule_based` |
| Risk-aware advice ("how safe is this?") | `ar_trend` or `bootstrap` + `rule_based`, report `p_feasible` |
| Use the data more fully | add `ml_yield`, **only if** its held-out MAE beats the baseline MAE on real data |
| Decide between weather models | `--compare`; prefer the one whose `cov80` is nearest 0.8 with comparable MAE |

Suggested order of work for the team:

1. Run the baseline on the real data; confirm the units and the printed summary.
2. Replace the fake crop conditions with sourced data.
3. Check the season-to-month mapping against agronomic reality.
4. Evaluate `ml_yield` on real data; keep it only if it earns its place.
5. Replace seasonal weather with real monthly climatology.
6. Add the economic ranking stage on top of the feasible set.

---

## 12. FAQ and glossary

**Why are some months identical (for example Nov and Dec)?** The data only has seasonal values. All months covered by the same seasons get the same number. Real monthly data would fix this.

**Why does the baseline forecast extend a trend, not predict actual weather?** No model here has any atmospheric information; the data cannot support it. These are climatology-style models.

**Why does nothing come out feasible?** Usually the dry-season months have little rain and the irrigation assumption (`--irrigation`, default 350 mm per season, a placeholder) is too low for the crops' water needs. Try a higher value, or check the precipitation unit guess.

**Why is `ml_yield` slower?** It trains a model on every run. Training happens once per run and is then reused for all samples.

**Why does a crop that is "grown here" sometimes lose to one that is not?** Because only water and temperature are hard rules, the local-history factor is a soft penalty (0.5 for never-seen), and the probability threshold can exclude the better-scoring crop (Section 7).

**Glossary**
- **Kharif / Rabi / Summer:** the main Indian cropping seasons (monsoon season, winter season, hot dry season).
- **Residual:** the difference between an actual value and the trend line.
- **AR(1):** a model in which each year's deviation depends partly on the previous year's.
- **Bootstrap (resampling):** building new scenarios by reusing real past data at random.
- **MAE:** mean absolute error; the average size of a prediction error.
- **Coverage (cov80):** the share of real values that fall inside the model's 10 to 90% range.
- **Held-out / time-based split:** testing on later years that the model never trained on.
- **Gradient boosting:** a model made of many small decision trees added one after another, each correcting the previous ones.
- **Feasible:** passes the hard water and temperature rules.
- **Effective rain:** the share of rainfall that reaches the crop (fixed at 80% here).

---

## 13. Repo map

```
imp/
  crop_reco.py            entry point
  README.md               short quick start
  TEAM_GUIDE.md           this document
  CODE_WALKTHROUGH.md     line-by-line notes on the ORIGINAL single script (legacy/)
  legacy/crop_reco_v1.py  the original baseline, kept for reference and as a test oracle
  krishi/
    config.py             constants: season mappings, variable names
    data.py               load and clean the dataset; unit guess; region choice
    units.py              rainfall unit conversion
    registry.py           plug-in registries (WEATHER_MODELS, SCORERS)
    crops.py              fake crop conditions; crop-name matching; yield priors
    pipeline.py           builds the contexts; forecast bands
    cli.py                command line, single run and --compare
    weather/
      forecaster.py       interface, shared machinery, backtest
      linear_trend.py     model 1 (baseline)
      ar_trend.py         model 2
      bootstrap.py        model 3
    scoring/
      scorer.py           interface
      rule_based.py       scorer 1 (baseline)
      ml_yield.py         scorer 2
      aggregate.py        probabilities and risk-aware ranking
  tests/
    mock_data.py          synthetic dataset with the real schema
    test_smoke.py         smoke tests
  output/                 generated CSVs
```
