import numpy as np
import pandas as pd

from config import MODELS

def _apparent_temp_c(t_c: float, rh: float, wind_kt: float) -> float:
    """
    Australian BOM apparent temperature formula — works for all climates.

    AT = T + 0.33*e − 0.70*ws − 4.00
    where
        e  = water-vapour pressure (kPa)
           = (rh/100) * 6.105 * exp(17.27*T / (237.7 + T))
        ws = wind speed (m/s)  [1 kt = 0.5144 m/s]

    Unlike heat-index (only valid >27°C) or wind-chill (only valid <10°C),
    this formula is continuous and valid across the full temperature range,
    making it suitable for both tropical and temperate cities.
    """
    ws = wind_kt * 0.5144                     # knots → m/s
    e  = (rh / 100.0) * 6.105 * np.exp(17.27 * t_c / (237.7 + t_c))
    return t_c + 0.33 * e - 0.70 * ws - 4.00


def safe_fillna(df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    """
    Fill NaN column-by-column. Falls back to 0.0 when fill_value is itself NaN.

    WHY: rolling features (forecast_anomaly, bias_14d_*) are NaN for the first
    few rows of any training fold. When the fold is small, the entire column can
    be NaN, making pandas .mean() return NaN, and fillna(NaN) is a no-op.
    Ridge then crashes with "Input X contains NaN".
    Zero is safe here because all affected features are mean-centred anomalies.
    """
    out = df.copy()
    for col in out.columns:
        fv = fill_values[col] if col in fill_values.index else np.nan
        if pd.isna(fv):
            fv = 0.0
        out[col] = out[col].fillna(fv)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from the paired METAR + model DataFrame.

    Feature groups
    --------------
    A) NWP raw temperatures        — individual model daily-max columns
    B) NWP ensemble stats          — spread, mean, std (model disagreement)
    C) Observed lags               — tmax yesterday/2-days/trend (persistence)
    D) Extended lags               — 5-day mean, 7-day anomaly (regime context)
    E) Per-model rolling bias      — 3-day (fast) and 14-day (slow drift)
    F) Forecast anomaly            — today's NWP vs recent 30-day climatology
    G) Cloud / solar               — cloud cover, effective solar, clear-calm index
    H) Moisture                    — dewpoint, dewpoint depression, humidity
    I) Moisture trends             — 3-day rolling dewpoint / humidity forecasts
    J) Wind direction              — forecast U/V components (city-specific effects)
    K) Apparent temperature        — heat island / sea-breeze signal from METAR
    L) Calendar                    — day-of-year, month (seasonal shape)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    tm = [f"temperature_2m_{m}_max" for m in MODELS
          if f"temperature_2m_{m}_max" in df.columns]

    # ── C) Observed lags ──────────────────────────────────────────────────────
    df["tmax_yesterday"]  = df["tmax_actual"].shift(1)
    df["tmax_2days_ago"]  = df["tmax_actual"].shift(2)
    df["temp_trend_3day"] = df["tmax_actual"].rolling(3).mean().shift(1)

    # ── D) Extended lags ─────────────────────────────────────────────────────
    # 5-day mean: captures medium-term synoptic regime better than 3-day
    df["tmax_5day_mean"]  = df["tmax_actual"].rolling(5,  min_periods=3).mean().shift(1)
    # 7-day anomaly: how much warmer/colder is the current regime vs last 30 days
    clim_30 = df["tmax_actual"].rolling(30, min_periods=10).mean().shift(1)
    df["tmax_7day_anomaly"] = (
        df["tmax_actual"].rolling(7, min_periods=3).mean().shift(1) - clim_30
    )

    # ── F) Forecast anomaly ───────────────────────────────────────────────────
    if "model_mean_max" in df.columns:
        clim = df["model_mean_max"].rolling(30, min_periods=7).mean()
        df["forecast_anomaly"] = df["model_mean_max"] - clim

    # ── B) Model spread ───────────────────────────────────────────────────────
    if len(tm) >= 2:
        df["model_max_spread"] = df[tm].max(axis=1) - df[tm].min(axis=1)

    # ── G) Cloud / solar ─────────────────────────────────────────────────────
    if "cloud_cover_forecast" in df.columns and "wind_forecast" in df.columns:
        df["clear_calm_index"] = (
            (100 - df["cloud_cover_forecast"]) / (df["wind_forecast"] + 1)
        )
    if "solar_ghi_forecast" in df.columns and "humidity_forecast" in df.columns:
        # Classic effective solar
        df["effective_solar"] = (
            df["solar_ghi_forecast"] * (1 - df["humidity_forecast"] / 200)
        )
    if "solar_ghi_forecast" in df.columns and "cloud_cover_forecast" in df.columns:
        # Physically more accurate: attenuate by cloud fraction directly
        df["solar_efficiency"] = (
            df["solar_ghi_forecast"] * (1 - df["cloud_cover_forecast"] / 100)
        )

    # ── E) Per-model rolling bias ─────────────────────────────────────────────
    for c in tm:
        err = df["tmax_actual"] - df[c]
        key = c.replace("temperature_2m_", "").replace("_max", "")
        df[f"bias_3d_{key}"]  = err.rolling(3,  min_periods=2).mean().shift(1)
        df[f"bias_14d_{key}"] = err.rolling(14, min_periods=3).mean().shift(1)

    # ── H) Moisture: dewpoint depression & humidity ───────────────────────────
    # Dewpoint depression (temp − dewpoint) is a direct measure of atmospheric
    # dryness.  A large depression → dry air → higher daytime tmax.
    # A small depression → humid / near-saturation → suppressed tmax.
    if "dewpoint_forecast" in df.columns and "model_mean_max" in df.columns:
        df["dewpoint_depression"] = df["model_mean_max"] - df["dewpoint_forecast"]
    elif "dewpoint_forecast" in df.columns:
        # Fall back to yesterday's tmax as a proxy for today's mean
        df["dewpoint_depression"] = df["tmax_yesterday"] - df["dewpoint_forecast"]

    # Observed dewpoint depression from METAR (even more local than NWP)
    if "dewpoint_at_peak" in df.columns:
        df["obs_dewpoint_depression"] = df["tmax_actual"] - df["dewpoint_at_peak"]
    # Lag it so it can be used to predict tomorrow
    if "obs_dewpoint_depression" in df.columns:
        df["obs_dewpoint_depression_lag1"] = df["obs_dewpoint_depression"].shift(1)

    # ── I) Moisture trends ────────────────────────────────────────────────────
    # 3-day rolling mean of forecast dewpoint — captures onset of moist/dry regimes
    # (monsoon onset, cold-front dryline passage) before tmax shift is fully visible
    if "dewpoint_forecast" in df.columns:
        df["dewpoint_trend_3day"] = (
            df["dewpoint_forecast"].rolling(3, min_periods=2).mean().shift(1)
        )
    if "humidity_forecast" in df.columns:
        df["humidity_trend_3day"] = (
            df["humidity_forecast"].rolling(3, min_periods=2).mean().shift(1)
        )

    # ── J) Wind direction ─────────────────────────────────────────────────────
    # NWP forecast wind direction as U/V components (already computed in
    # load_model_forecasts as ens_wind_u / ens_wind_v, propagated through
    # daily_model as wind_u_forecast / wind_v_forecast).
    # These let the model learn directional effects per city without encoding
    # the direction by hand (e.g. sea-breeze vs land-breeze in Mumbai, onshore
    # vs offshore in Busan, Föhn vs Bise in Zurich).
    # No transformation needed — they arrive as [-1, 1] normalised components.

    # ── K) Apparent temperature / heat island signal ───────────────────────────
    # apparent_temp_peak is computed in daily_metar from tmpc + relh + wind_kt.
    # heat_island_signal = apparent − dry-bulb: positive means humidity/low-wind
    # amplifies warmth (urban heat island, still humid days); negative means
    # wind chill / evaporative cooling reduces felt temperature.
    if "apparent_temp_peak" in df.columns:
        df["heat_island_signal"] = df["apparent_temp_peak"] - df["tmax_actual"]
    # Lag so it can be used for tomorrow's forecast
    if "heat_island_signal" in df.columns:
        df["heat_island_signal_lag1"] = df["heat_island_signal"].shift(1)

    # ── L) Calendar ───────────────────────────────────────────────────────────
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"]       = df["date"].dt.month

    # Drop only on anchor features — bias/anomaly NaNs handled by safe_fillna
    df = df.dropna(subset=["tmax_yesterday", "temp_trend_3day"]).reset_index(drop=True)
    print(f"  [FEAT]  {len(df)} rows x {len(df.columns)} cols")
    return df

def feature_cols(df: pd.DataFrame) -> list:
    """
    Return the ordered list of feature columns that exist in df.

    Order follows the importance hierarchy: NWP temperatures first (dominant
    signal), then ensemble stats, then observed lags, then bias trackers, then
    atmospheric corrections, then moisture/wind/apparent-temp additions, then
    calendar.  GBM is order-agnostic but Ridge benefits from correlated features
    being grouped — keeping NWP columns together reduces multicollinearity noise.
    """
    c = []

    # A) Individual NWP model temperature maxima
    for m in MODELS:
        col = f"temperature_2m_{m}_max"
        if col in df.columns:
            c.append(col)

    # B) Ensemble stats (model disagreement / spread)
    for col in ["model_spread", "model_mean_max", "model_std_max", "model_max_spread"]:
        if col in df.columns:
            c.append(col)

    # C+D) Observed lags (short, medium, regime-level)
    for col in ["tmax_yesterday", "tmax_2days_ago", "temp_trend_3day",
                "tmax_5day_mean", "tmax_7day_anomaly"]:
        if col in df.columns:
            c.append(col)

    # E) Per-model short-window bias (fast — regime shifts within 3 days)
    for m in MODELS:
        col = f"bias_3d_{m}"
        if col in df.columns:
            c.append(col)

    # E) Per-model long-window bias (slow — seasonal model drift)
    for m in MODELS:
        col = f"bias_14d_{m}"
        if col in df.columns:
            c.append(col)

    # F) Forecast anomaly vs 30-day climatology
    for col in ["forecast_anomaly"]:
        if col in df.columns:
            c.append(col)

    # G) Cloud / solar
    for col in ["cloud_cover_forecast", "solar_ghi_forecast",
                "clear_calm_index", "effective_solar", "solar_efficiency"]:
        if col in df.columns:
            c.append(col)

    # H) Moisture: humidity + dewpoint features
    for col in ["humidity_forecast", "dewpoint_forecast",
                "dewpoint_depression", "obs_dewpoint_depression_lag1"]:
        if col in df.columns:
            c.append(col)

    # I) Moisture trends (3-day rolling — regime-onset signal)
    for col in ["dewpoint_trend_3day", "humidity_trend_3day"]:
        if col in df.columns:
            c.append(col)

    # J) Wind: speed + direction components
    for col in ["wind_forecast", "wind_u_forecast", "wind_v_forecast"]:
        if col in df.columns:
            c.append(col)

    # K) Apparent temperature / heat island (METAR-derived)
    for col in ["heat_island_signal_lag1"]:
        if col in df.columns:
            c.append(col)

    # L) Calendar
    for col in ["day_of_year", "month"]:
        if col in df.columns:
            c.append(col)

    return c