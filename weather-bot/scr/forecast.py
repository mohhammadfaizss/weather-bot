import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import datetime
from scipy.stats import norm
from config import MODELS
from features import feature_cols, safe_fillna
from data import get_tomorrow_model_row
from corrector import RealtimeBiasCorrector
from model import predict_row

def get_target_date(paired_df: pd.DataFrame,
                    model_daily_df: pd.DataFrame = None) -> str:
    """
    Derive the target prediction date from the paired training data.

    target_date = last date in paired_df + 1 day.

    If model_daily_df is supplied, we verify that tomorrow's row actually exists
    in the model data and warn clearly if it does not — this means your model
    CSV does not yet include tomorrow's forecast and the pipeline cannot predict.

    paired_df["date"] is already in local time (daily_metar / daily_model both
    convert to local), so no clock or timezone arithmetic is needed here.
    """
    last   = paired_df["date"].max()
    target = last + datetime.timedelta(days=1)
    target_str = target.strftime("%Y-%m-%d")

    if model_daily_df is not None:
        model_max = model_daily_df["date"].max()
        if model_max < target:
            print(f"  [WARNING] Model data ends at {model_max.date()} but target is "
                  f"{target_str}. Tomorrow's NWP forecast is missing — "
                  f"make sure your model CSV includes tomorrow's data.")

    return target_str


def make_forecast(model, scaler, use_scaling, fill_values, fc,
                  fd, mo_raw, timezone, target_date, corrector, error_series):
    """
    Build a synthetic "tomorrow" row and predict its temperature.

    WHY THIS IS NECESSARY
    ---------------------
    The paired DataFrame (fd) only contains dates where BOTH METAR and model
    data exist. Since METAR has no tomorrow, fd ends at today. If we used
    fd.iloc[-1] directly we would be predicting today not tomorrow — the model
    features would be today's NWP forecasts, not tomorrow's.

    THE CORRECT APPROACH
    --------------------
    Two separate data sources are needed:
      - Model columns  : tomorrow's row in od (the daily_model output).
                         This is the date fd ends at + 1. It exists in od
                         because NWP models always forecast ahead.
      - Lag features   : computed from fd (METAR-based actuals up to today).
                         tmax_yesterday, tmax_2days_ago, temp_trend_3day,
                         and all bias_* features come from here.
      - Derived features: recomputed from the tomorrow model row + today's lags.
                         forecast_anomaly, model_spread, clear_calm_index, etc.

    Parameters
    ----------
    fd          : paired + featured DataFrame, ends at today
    od          : daily_model output (model-only daily rows), includes tomorrow
    target_date : str "YYYY-MM-DD" — the date being predicted (tomorrow)
    """
    import warnings

    # ── 1. Get tomorrow's model row from raw hourly data ─────────────────────
    # get_tomorrow_model_row() bypasses the >= 20 obs filter so it works even
    # when tomorrow has only a few hours of data in the model download window.
    # target_ts is used later for calendar features.
    target_ts = pd.Timestamp(target_date)
    tom       = get_tomorrow_model_row(mo_raw, timezone, target_date)
    today     = fd.iloc[-1]
    # ── 2. Build the synthetic forecast row ──────────────────────────────────
    # Start from a clean dict so we have full control over every feature.
    # Three categories of features need different sources:
    #
    #   A) Tomorrow's NWP values  → from tom (tomorrow's model row in od)
    #   B) Lag / bias features    → from fd (today's paired row — METAR-derived)
    #   C) Derived / calendar     → computed fresh
    row = tom.copy()
    today = fd.iloc[-1]

    # ── A) Observed lag features (from METAR actuals up to today) ─────────────
    # build_features defines these via .shift() on tmax_actual:
    #   tmax_yesterday  = shift(1) → for tomorrow's row = today's actual
    #   tmax_2days_ago  = shift(2) → yesterday's actual
    #   temp_trend_3day = rolling(3).mean().shift(1) → 3-day mean up to today
    actuals = fd["tmax_actual"]
    row["tmax_yesterday"]  = float(actuals.iloc[-1])
    row["tmax_2days_ago"]  = float(actuals.iloc[-2]) if len(actuals) >= 2 else float(actuals.iloc[-1])
    row["temp_trend_3day"] = float(actuals.iloc[-3:].mean())
    row["tmax_5day_mean"]  = float(actuals.iloc[-5:].mean())

    # 7-day anomaly: (7-day mean) - (30-day mean), both ending at today
    mean_7d  = float(actuals.iloc[-7:].mean())
    mean_30d = float(actuals.iloc[-30:].mean()) if len(actuals) >= 10 else mean_7d
    row["tmax_7day_anomaly"] = mean_7d - mean_30d

    # ── B) Rolling bias, anomaly, moisture-trend features ─────────────────────
    # These are all computed from historical rows in fd — tomorrow has no actual
    # yet so we carry today's computed values forward unchanged.
    carry_prefixes = ("bias_3d_", "bias_14d_", "dewpoint_trend_3day",
                      "humidity_trend_3day")
    carry_exact    = {"forecast_anomaly", "obs_dewpoint_depression_lag1",
                      "heat_island_signal_lag1"}
    for col in fc:
        is_carry = (any(col.startswith(p) for p in carry_prefixes)
                    or col in carry_exact)
        if is_carry and col in today.index and pd.notna(today[col]):
            row[col] = today[col]

    # obs_dewpoint_depression_lag1: carry today's observed depression
    # (it's already lag-1 relative to tomorrow — perfect as-is)
    if "obs_dewpoint_depression" in today.index and pd.notna(today["obs_dewpoint_depression"]):
        row["obs_dewpoint_depression_lag1"] = float(today["obs_dewpoint_depression"])

    # heat_island_signal_lag1: carry today's METAR heat-island signal
    if "heat_island_signal" in today.index and pd.notna(today["heat_island_signal"]):
        row["heat_island_signal_lag1"] = float(today["heat_island_signal"])

    # ── C) Derived features recomputed from tomorrow's NWP values ─────────────
    tm_cols = [f"temperature_2m_{m}_max" for m in MODELS
               if f"temperature_2m_{m}_max" in row.index]
    if len(tm_cols) >= 2:
        vals = [row[c] for c in tm_cols if pd.notna(row.get(c))]
        row["model_max_spread"] = max(vals) - min(vals) if len(vals) >= 2 else 0.0

    cloud    = row.get("cloud_cover_forecast", 50.0)
    wind     = row.get("wind_forecast",        5.0)
    solar    = row.get("solar_ghi_forecast",   0.0)
    humidity = row.get("humidity_forecast",    70.0)
    dewpoint = row.get("dewpoint_forecast",    float("nan"))

    row["clear_calm_index"] = (100 - cloud) / (wind + 1)
    row["effective_solar"]  = solar * (1 - humidity / 200)
    row["solar_efficiency"] = solar * (1 - cloud / 100)

    # Dewpoint depression: tomorrow's NWP mean temp minus tomorrow's dewpoint
    model_mean = row.get("model_mean_max", float("nan"))
    if not np.isnan(dewpoint):
        ref_temp = model_mean if not np.isnan(model_mean) else row["tmax_yesterday"]
        row["dewpoint_depression"] = ref_temp - dewpoint

    # Dewpoint trend: 3-day rolling of *forecast* dewpoint ending at today,
    # then apply tomorrow's value to push the trend one step forward
    if "dewpoint_forecast" in fd.columns and not np.isnan(dewpoint):
        hist_dp = list(fd["dewpoint_forecast"].iloc[-2:])   # last 2 days
        hist_dp.append(dewpoint)                             # add tomorrow
        row["dewpoint_trend_3day"] = float(np.mean([v for v in hist_dp
                                                     if not np.isnan(v)]))
    if "humidity_forecast" in fd.columns and not np.isnan(humidity):
        hist_rh = list(fd["humidity_forecast"].iloc[-2:])
        hist_rh.append(humidity)
        row["humidity_trend_3day"] = float(np.mean([v for v in hist_rh
                                                     if not np.isnan(v)]))

    # ── D) Calendar features for tomorrow's date ──────────────────────────────
    row["day_of_year"] = target_ts.dayofyear
    row["month"]       = target_ts.month

    # ── 3. Predict ───────────────────────────────────────────────────────────
    ml_p          = predict_row(model, scaler, use_scaling, fill_values, fc, row)
    last_observed = float(fd["tmax_actual"].iloc[-1])   # today's observed tmax

    rt    = corrector.correction(model_forecast=ml_p, last_observed=last_observed)
    final = round(ml_p + rt, 1)

    es = float(error_series.std())
    eb = float(error_series.mean())

    return {
        "final_forecast":      final,
        "ml_forecast":         round(ml_p, 1),
        "last_observed_tmax":  last_observed,
        "forecast_jump":       round(ml_p - last_observed, 1),
        "realtime_correction": round(rt, 2),
        "corrector_n_days":    corrector.n_days(),
        "corrector_summary":   corrector.summary(
                                   model_forecast=ml_p,
                                   last_observed=last_observed),
        "insample_bias":       round(eb, 2),
        "error_std":           round(es, 2),
        "ci_80": (round(final - 1.28*es, 1), round(final + 1.28*es, 1)),
        "ci_95": (round(final - 1.96*es, 1), round(final + 1.96*es, 1)),
    }
# def bucket_probs(forecast, error_std, buckets):

#     probs = {}
#     for i, b in enumerate(buckets):
#         if   b == buckets[0]:  p = norm.cdf((buckets[i+1]+b)/2, forecast, error_std)
#         elif b == buckets[-1]: p = 1 - norm.cdf((b+buckets[i-1])/2, forecast, error_std)
#         else:
#             lo = (b+buckets[i-1])/2; hi = (buckets[i+1]+b)/2
#             p  = norm.cdf(hi, forecast, error_std) - norm.cdf(lo, forecast, error_std)
#         probs[b] = round(float(p), 4)
#     return probs

# def bet_recs(ml_probs, market_prices, min_edge=0.05):
#     recs = []
#     for b, mp in ml_probs.items():
#         if b not in market_prices: continue
#         mkt  = market_prices[b]; edge = mp - mkt
#         if abs(edge) >= min_edge:
#             kelly = edge/(1-mkt) if edge>0 else edge/mkt
#             recs.append({"bucket": b, "ml_prob": f"{mp:.1%}",
#                          "market_price": f"{mkt:.1%}", "edge": f"{edge:+.1%}",
#                          "kelly": f"{kelly:.3f}",
#                          "action": "BET YES" if edge>0 else "BET NO",
#                          "confidence": "HIGH" if abs(edge)>0.10 else "MEDIUM"})
#     recs.sort(key=lambda x: abs(float(x["edge"].replace("%",""))/100), reverse=True)
#     return recs
