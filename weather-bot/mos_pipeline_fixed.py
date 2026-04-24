"""
MOS (Model Output Statistics) Pipeline  v3
==========================================
General-purpose: any city / climate / timezone.

KEY CHANGES vs previous versions
---------------------------------
1. NaN crash fixed via safe_fillna() — median imputation with 0.0 fallback
2. RealtimeBiasCorrector seeded from WALK-FORWARD out-of-sample errors,
   not in-sample predictions (which are near-zero by definition)
3. Walk-forward is always run for at least the last 14 days so the corrector
   has real signal even when full walk-forward is disabled for speed
4. Corrector state is printed so you can verify it is working

DATA FORMATS
------------
historical_<City>.csv  or  historical.csv:
  city, date (UTC), temperature_2m_<model>, relative_humidity_2m_<model>,
  shortwave_radiation_<model>, cloud_cover_low/mid/high_<model>, wind_speed_10m_<model>

<STATION>.csv  (Iowa State ASOS, UTC timestamps):
  station, valid, tmpc, dwpc, relh, skyc1, sknt, skyc2, skyc3, metar

TIMEZONE
--------
Always pass an IANA string  e.g. "Asia/Kolkata", "Europe/Madrid"
Never pass a fixed offset — it silently breaks DST cities.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
from variable import cities

try:
    import pytz
except ImportError:
    raise ImportError("pip install pytz")

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = [
    "ecmwf_ifs",
    "ecmwf_ifs025",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless",
    "ukmo_seamless",
]

SKY_MAP = {"NSC": 0, "FEW": 20, "SCT": 45, "BKN": 75, "OVC": 100}


# =============================================================================
# SAFE IMPUTATION
# =============================================================================

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


# =============================================================================
# REAL-TIME BIAS CORRECTOR
# =============================================================================

class RealtimeBiasCorrector:
    """
    Short-memory correction layer on top of the ML model.

    WHY IT EXISTS
    -------------
    The ML model learns a climatological correction from years of data.
    But forecast errors are regime-dependent: if the model has been running
    2C warm for the last 5 days, it will almost certainly do so tomorrow too.
    The ML model cannot capture this because it changes faster than any
    training window.

    WEIGHTING
    ---------
    Uses exponential decay so recent days matter more than older ones.
    The decay rate is tunable:
        decay=0.5  very aggressive — most recent day dominates (too jumpy)
        decay=0.7  moderate — good default, smooth but still recency-aware
        decay=0.9  nearly equal weights — best for very stable slow biases

    With 5 days and decay=0.7, the effective weights are roughly:
        day-1: 36%,  day-2: 26%,  day-3: 18%,  day-4: 13%,  day-5: 7%

    JUMP DAMPENING
    --------------
    When tomorrow's model forecast is significantly different from the last
    observed tmax, the recent error history is from a different weather regime
    and should not be trusted. A 2C+ forecast jump means a regime transition
    is likely in progress — the corrector damps its output proportionally.

    The dampening uses an exponential decay on jump size:
        dampening = exp(-jump / jump_scale)
        jump_scale=2.0 → 1C jump keeps 61%, 2C keeps 37%, 3C keeps 22%,
                          4C keeps 13%, 5C+ is near zero correction

    This prevents the corrector from pushing the forecast in the wrong
    direction on heat spike or cold outbreak days.

    CRITICAL: must be seeded from OUT-OF-SAMPLE walk-forward errors,
    not in-sample predictions (in-sample errors are near-zero by definition).

    PRODUCTION USE
    --------------
    Each day after observing the real tmax:
        corrector.update(date, ml_prediction, actual_tmax)
    Next morning before issuing the forecast:
        final = corrector.apply(ml_pred, last_observed_tmax)
    Persist between runs:
        import json
        json.dump(corrector.state(), open("corrector_state.json","w"), default=str)
        corrector = RealtimeBiasCorrector.from_state(
                        json.load(open("corrector_state.json")))
    """

    def __init__(self, window: int = 5, min_periods: int = 3,
                 decay: float = 0.7, jump_scale: float = 2.0):
        self.window      = window
        self.min_periods = min_periods
        self.decay       = decay       # exponential decay rate per day back
        self.jump_scale  = jump_scale  # scale for jump dampening (degrees C)
        self._history: list = []       # [(date, error), ...]

    def update(self, date, ml_prediction: float, actual: float) -> None:
        """Record error = actual - ml_prediction. Call AFTER observing METAR."""
        self._history.append((date, float(actual) - float(ml_prediction)))
        self._history = self._history[-self.window:]

    def correction(self, model_forecast: float = None,
                   last_observed: float = None) -> float:
        """
        Compute the bias correction to add to tomorrow's ML forecast.

        Parameters
        ----------
        model_forecast  : ML model's raw prediction for tomorrow.
                          When provided alongside last_observed, enables
                          jump dampening.
        last_observed   : The most recently observed actual tmax.
                          Used to measure the size of the forecast jump.

        Returns
        -------
        Correction in degrees C. Add this to the ML forecast.
        Returns 0.0 when fewer than min_periods days are available.
        """
        if len(self._history) < self.min_periods:
            return 0.0

        # --- Exponentially weighted mean of recent errors ---
        errors  = np.array([e for _, e in self._history])
        n       = len(errors)
        # Most recent day is index n-1; oldest is index 0
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        base = float(np.dot(weights, errors))

        # --- Jump dampening ---
        # When the model predicts a large departure from recent observed temps,
        # the recent error history is stale (different regime). Damp the
        # correction proportionally so we trust the models during transitions.
        if model_forecast is not None and last_observed is not None:
            jump       = abs(float(model_forecast) - float(last_observed))
            dampening  = float(np.exp(-jump / self.jump_scale))
            corrected  = base * dampening
            return round(corrected, 2)

        return round(base, 2)

    def n_days(self) -> int:
        return len(self._history)

    def apply(self, ml_prediction: float,
              last_observed: float = None) -> float:
        """Return bias-corrected forecast."""
        corr = self.correction(
            model_forecast=ml_prediction,
            last_observed=last_observed
        )
        return round(float(ml_prediction) + corr, 1)

    def summary(self, model_forecast: float = None,
                last_observed: float = None) -> str:
        if not self._history:
            return "no history"
        errs   = [e for _, e in self._history]
        base   = self.correction()   # without dampening
        damped = self.correction(model_forecast, last_observed)
        jump   = (abs(float(model_forecast) - float(last_observed))
                  if model_forecast is not None and last_observed is not None
                  else None)
        parts  = [
            f"{len(errs)} days",
            f"decay={self.decay}",
            f"errors: [{', '.join(f'{e:+.1f}' for e in errs)}]",
            f"base correction: {base:+.2f}C",
        ]
        if jump is not None:
            damp_pct = 100 * (1 - abs(damped) / abs(base)) if base != 0 else 0
            parts.append(f"forecast jump: {jump:.1f}C  "
                         f"dampening: {damp_pct:.0f}%  "
                         f"final: {damped:+.2f}C")
        return "  |  ".join(parts)

    def state(self) -> dict:
        return {"window":      self.window,
                "min_periods": self.min_periods,
                "decay":       self.decay,
                "jump_scale":  self.jump_scale,
                "history":     [(str(d), e) for d, e in self._history]}

    @classmethod
    def from_state(cls, state: dict) -> "RealtimeBiasCorrector":
        obj = cls(window     = state["window"],
                  min_periods= state["min_periods"],
                  decay      = state.get("decay",      0.7),
                  jump_scale = state.get("jump_scale", 2.0))
        obj._history = list(state["history"])
        return obj


# =============================================================================
# TIMEZONE
# =============================================================================

def to_local(series: pd.Series, tz_name: str) -> pd.Series:
    """DST-correct UTC -> local. Never use a fixed offset for DST cities."""
    tz = pytz.timezone(tz_name)
    s  = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)


# =============================================================================
# STAGE 1: LOAD
# =============================================================================

def load_metar(station: str, data_folder: str = "mos_data") -> pd.DataFrame:
    df = pd.read_csv(Path(data_folder) / f"{station}.csv")
    df["valid"] = pd.to_datetime(df["valid"], utc=False)
    if "sknt" in df.columns and "wind_kt" not in df.columns:
        df = df.rename(columns={"sknt": "wind_kt"})
    return df


def load_model_forecasts(city: str, data_folder: str = "mos_data") -> pd.DataFrame:
    p1 = Path(data_folder) / f"historical_{city}.csv"
    p2 = Path(data_folder) / "historical.csv"
    path = p1 if p1.exists() else (p2 if p2.exists() else None)
    if path is None:
        raise FileNotFoundError(f"No model data found. Tried {p1} and {p2}")

    df = pd.read_csv(path)
    df = df[df["city"].str.lower() == city.lower()].copy()
    if len(df) == 0:
        raise ValueError(f"No rows for city='{city}' in {path}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    for m in MODELS:
        lo = df.get(f"cloud_cover_low_{m}",  pd.Series(0, index=df.index))
        mi = df.get(f"cloud_cover_mid_{m}",  pd.Series(0, index=df.index))
        hi = df.get(f"cloud_cover_high_{m}", pd.Series(0, index=df.index))
        df[f"cloud_cover_{m}"] = (lo + mi + hi).clip(upper=100)

    for prefix, out in [("temperature_2m",       "ens_temp"),
                         ("relative_humidity_2m", "ens_humidity"),
                         ("shortwave_radiation",  "ens_solar"),
                         ("wind_speed_10m",       "ens_wind")]:
        cols = [f"{prefix}_{m}" for m in MODELS if f"{prefix}_{m}" in df.columns]
        if cols: df[out] = df[cols].mean(axis=1)

    cc = [f"cloud_cover_{m}" for m in MODELS if f"cloud_cover_{m}" in df.columns]
    if cc: df["ens_cloud"] = df[cc].mean(axis=1)

    return df.sort_values("date").reset_index(drop=True)


# =============================================================================
# STAGE 2: CLEAN
# =============================================================================

def clean_metar(df, tmin=-40., tmax=60.):
    n = len(df)
    df = df[df["tmpc"].between(tmin, tmax)].copy()
    df = df.drop_duplicates(subset=["station", "valid"])
    df = df.sort_values(["station", "valid"]).reset_index(drop=True)
    df["td"] = df.groupby("station")["tmpc"].diff().abs()
    bad = df["td"] > 5
    if bad.sum(): print(f"  [CLEAN] {bad.sum()} suspicious METAR jumps flagged")
    df = df[~bad].drop(columns=["td"])
    print(f"  [CLEAN] METAR: {n} -> {len(df)}")
    return df


def clean_model(df, tmin=-40., tmax=60.):
    n     = len(df)
    tcols = [f"temperature_2m_{m}" for m in MODELS if f"temperature_2m_{m}" in df.columns]
    df    = df.dropna(subset=tcols, how="all").copy()
    mask  = pd.DataFrame({c: df[c].between(tmin, tmax) for c in tcols}).any(axis=1)
    df    = df[mask]
    print(f"  [CLEAN] Model: {n} -> {len(df)}")
    return df


# =============================================================================
# STAGE 3: ALIGN AND PAIR
# =============================================================================

def daily_metar(metar_df, station, tz):
    df = metar_df[metar_df["station"] == station].copy()
    df["lt"] = to_local(df["valid"], tz)
    df["ld"] = df["lt"].dt.date
    rows = []
    for d, g in df.groupby("ld"):
        if len(g) < 20: continue
        idx      = g["tmpc"].idxmax()
        peak     = g[g["lt"].dt.hour.between(11, 15)]
        if len(peak) == 0: peak = g
        cloud    = peak["skyc1"].map(SKY_MAP).fillna(50).mean()
        rows.append({
            "date":             pd.Timestamp(d),
            "tmax_actual":      round(float(g.loc[idx, "tmpc"]), 1),
            "tmax_hour_local":  int(g.loc[idx, "lt"].hour),
            "wind_at_peak":     round(float(peak["wind_kt"].mean()), 1),
            "humidity_at_peak": round(float(peak["relh"].mean()), 1),
            "cloud_at_peak":    round(float(cloud), 1),
            "obs_count":        len(g),
        })
    r = pd.DataFrame(rows)
    print(f"  [ALIGN] {len(r)} daily METAR records ({station})")
    return r


def daily_model(model_df, tz):
    df    = model_df.copy()
    df["lt"] = to_local(df["date"].dt.tz_localize("UTC"), tz)
    df["ld"] = df["lt"].dt.date
    tcols = [f"temperature_2m_{m}" for m in MODELS if f"temperature_2m_{m}" in df.columns]
    rows  = []
    for d, g in df.groupby("ld"):
        if len(g) < 20: continue
        row = {"date": pd.Timestamp(d)}
        for c in tcols: row[f"{c}_max"] = float(g[c].max())
        mx = [row[f"{c}_max"] for c in tcols]
        row["model_spread"]   = float(max(mx) - min(mx))
        row["model_mean_max"] = float(np.mean(mx))
        row["model_std_max"]  = float(np.std(mx))
        pk = g[g["lt"].dt.hour.between(11, 15)]
        if len(pk) == 0: pk = g
        for src, dst in [("ens_cloud",    "cloud_cover_forecast"),
                         ("ens_humidity", "humidity_forecast"),
                         ("ens_wind",     "wind_forecast"),
                         ("ens_solar",    "solar_ghi_forecast")]:
            if src in df.columns: row[dst] = float(pk[src].mean())
        rows.append(row)
    r = pd.DataFrame(rows)
    print(f"  [ALIGN] {len(r)} daily model records")
    return r


def pair(metar_d, model_d):
    p = pd.merge(metar_d, model_d, on="date", how="inner")
    print(f"  [PAIR]  {len(p)} paired days")
    if len(p) == 0:
        raise ValueError("No overlapping dates — check timezone / city name / date ranges")
    return p


# =============================================================================
# STAGE 4: FEATURES
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    tm = [f"temperature_2m_{m}_max" for m in MODELS
          if f"temperature_2m_{m}_max" in df.columns]

    df["tmax_yesterday"]  = df["tmax_actual"].shift(1)
    df["tmax_2days_ago"]  = df["tmax_actual"].shift(2)
    df["temp_trend_3day"] = df["tmax_actual"].rolling(3).mean().shift(1)

    if "model_mean_max" in df.columns:
        clim = df["model_mean_max"].rolling(30, min_periods=7).mean()
        df["forecast_anomaly"] = df["model_mean_max"] - clim

    if len(tm) >= 2:
        # Inter-model spread only — no top/bottom aggregates (they double-count
        # individual model columns already in the feature set)
        df["model_max_spread"] = df[tm].max(axis=1) - df[tm].min(axis=1)

    if "cloud_cover_forecast" in df.columns and "wind_forecast" in df.columns:
        df["clear_calm_index"] = (100 - df["cloud_cover_forecast"]) / (df["wind_forecast"] + 1)
    if "solar_ghi_forecast" in df.columns and "humidity_forecast" in df.columns:
        df["effective_solar"]  = df["solar_ghi_forecast"] * (1 - df["humidity_forecast"] / 200)

    for c in tm:
        err = df["tmax_actual"] - df[c]
        key = c.replace("temperature_2m_", "").replace("_max", "")
        # 3-day: catches fast regime shifts (heat spikes, cold outbreaks)
        df[f"bias_3d_{key}"]  = err.rolling(3,  min_periods=2).mean().shift(1)
        # 14-day: captures slower seasonal model drift
        df[f"bias_14d_{key}"] = err.rolling(14, min_periods=3).mean().shift(1)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"]       = df["date"].dt.month

    # Only drop on anchor features — do NOT drop bias/anomaly NaNs here.
    # safe_fillna handles those at training time.
    df = df.dropna(subset=["tmax_yesterday", "temp_trend_3day"]).reset_index(drop=True)
    print(f"  [FEAT]  {len(df)} rows x {len(df.columns)} cols")
    return df


def feature_cols(df: pd.DataFrame) -> list:
    c = []
    for m in MODELS:
        col = f"temperature_2m_{m}_max"
        if col in df.columns: c.append(col)
    for col in ["model_spread","model_mean_max","model_std_max","model_max_spread",
                "cloud_cover_forecast","humidity_forecast","wind_forecast","solar_ghi_forecast",
                "clear_calm_index","effective_solar","forecast_anomaly",
                "tmax_yesterday","tmax_2days_ago","temp_trend_3day"]:
        if col in df.columns: c.append(col)
    # Short-window bias first — more responsive to recent regime changes
    for m in MODELS:
        col = f"bias_3d_{m}"
        if col in df.columns: c.append(col)
    # Long-window bias — slower seasonal model drift
    for m in MODELS:
        col = f"bias_14d_{m}"
        if col in df.columns: c.append(col)
    for col in ["day_of_year","month"]:
        if col in df.columns: c.append(col)
    return c


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def fit_model(X_raw: pd.DataFrame, y: pd.Series):
    """Fit GBM (>=60 rows) or Ridge. Returns (model, scaler, use_scaling, fill_values)."""
    fv = X_raw.median()                     # median fill — robust to skew
    X  = safe_fillna(X_raw, fv)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    if len(X) >= 60:
        m = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42)
        m.fit(X, y)
        return m, sc, False, fv
    else:
        m = Ridge(alpha=1.0)
        m.fit(Xs, y)
        return m, sc, True, fv


def predict_row(model, scaler, use_scaling, fill_values, fcols, row):
    X = safe_fillna(row[fcols].to_frame().T, fill_values)
    return float(model.predict(scaler.transform(X) if use_scaling else X)[0])


# =============================================================================
# STAGE 5a: SEED WALK-FORWARD
# Runs the last `seed_days` through expanding-window training to produce
# honest out-of-sample errors that seed the RealtimeBiasCorrector.
# This is fast (<30s for seed_days=14) and ALWAYS runs.
# =============================================================================

def seed_corrector(df: pd.DataFrame, seed_days: int = 14,
                   decay: float = 0.7, jump_scale: float = 2.0) -> RealtimeBiasCorrector:
    """
    Run expanding-window prediction for the last `seed_days` days.

    Each prediction is made by a model trained on ALL data BEFORE that day,
    so these are genuine out-of-sample errors — the model has never seen the
    day it is predicting.

    The corrector is then loaded with these real errors and is ready to apply
    a meaningful regime correction to tomorrow's forecast.
    """
    fc   = feature_cols(df)
    corr = RealtimeBiasCorrector(window=5, min_periods=3, decay=decay, jump_scale=jump_scale)
    n    = len(df)

    # We need enough training data before the first seed day.
    # Use at most the last seed_days rows, but ensure at least 60 training rows.
    min_train = 60
    start_idx = max(min_train, n - seed_days)

    if start_idx >= n:
        print(f"  [SEED]  Not enough data to seed corrector")
        return corr

    print(f"  [SEED]  Running {n - start_idx} day(s) of out-of-sample prediction "
          f"to seed real-time corrector...")

    for test_idx in range(start_idx, n):
        train = df.iloc[:test_idx]
        row   = df.iloc[test_idx]
        m, sc, us, fv = fit_model(train[fc], train["tmax_actual"])
        ml_p  = predict_row(m, sc, us, fv, fc, row)
        actual = float(row["tmax_actual"])
        corr.update(row["date"], ml_p, actual)

    print(f"  [SEED]  Corrector state: {corr.summary()}")
    return corr


# =============================================================================
# STAGE 5b: FULL WALK-FORWARD VALIDATION (optional, for diagnostics)
# =============================================================================

def walk_forward_validate(df: pd.DataFrame,
                           initial_train_days: int = 365) -> pd.DataFrame:
    """
    Full expanding-window validation with real-time bias correction.
    Produces honest out-of-sample MAE for all years after the first.
    Set run_walk_forward=False in run_pipeline to skip this for speed.
    """
    fc   = feature_cols(df)
    corr = RealtimeBiasCorrector(window=5, min_periods=3, decay=0.7, jump_scale=2.0)
    rows = []
    n    = len(df)

    if n <= initial_train_days:
        print(f"  [WF]  Skipped — need >{initial_train_days} rows, have {n}")
        return pd.DataFrame()

    total = n - initial_train_days
    print(f"  [WF]  {total} test days  |  initial train: {initial_train_days} days")

    for i, idx in enumerate(range(initial_train_days, n)):
        train  = df.iloc[:idx]
        row    = df.iloc[idx]
        m, sc, us, fv = fit_model(train[fc], train["tmax_actual"])
        ml_p   = predict_row(m, sc, us, fv, fc, row)
        rt     = corr.correction()
        final  = round(ml_p + rt, 1)
        actual = float(row["tmax_actual"])
        corr.update(row["date"], ml_p, actual)
        rows.append({"date": row["date"], "actual": actual,
                     "ml_pred": round(ml_p, 1), "rt_correction": round(rt, 1),
                     "final_pred": final,
                     "err_ml": round(actual - ml_p, 1),
                     "err_final": round(actual - final, 1)})
        if i % 100 == 0:
            print(f"    {i}/{total}  ({100*i/total:.0f}%)")

    r = pd.DataFrame(rows)
    mae_ml = r["err_ml"].abs().mean()
    mae_f  = r["err_final"].abs().mean()
    w1m = (r["err_ml"].abs()    <= 1).mean()*100
    w1f = (r["err_final"].abs() <= 1).mean()*100
    w2m = (r["err_ml"].abs()    <= 2).mean()*100
    w2f = (r["err_final"].abs() <= 2).mean()*100

    print(f"\n  Walk-forward ({len(r)} test days)")
    print(f"  {'':30s} {'ML':>8}  {'+ RT':>8}")
    print(f"  {'MAE':30s} {mae_ml:>7.2f}C  {mae_f:>7.2f}C")
    print(f"  {'Within +/-1C':30s} {w1m:>7.1f}%  {w1f:>7.1f}%")
    print(f"  {'Within +/-2C':30s} {w2m:>7.1f}%  {w2f:>7.1f}%")
    return r


# =============================================================================
# STAGE 5c: QUICK CV  (diagnostic only)
# =============================================================================

def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    fc       = feature_cols(df)
    fold_sz  = len(df) // (n_splits + 1)
    ml_l, mg = [], []

    print(f"\n  {'Fold':<6} {'Train':<8} {'Test':<8} {'Ridge':>8}  {'GBM':>8}")
    print("  " + "-" * 46)

    for fold in range(n_splits):
        t_end  = fold_sz * (fold + 1)
        te_end = min(t_end + fold_sz, len(df))
        tr     = df.iloc[:t_end]
        te     = df.iloc[t_end:te_end]
        if len(te) == 0: continue

        fv = tr[fc].median()
        Xtr = safe_fillna(tr[fc], fv);  ytr = tr["tmax_actual"]
        Xte = safe_fillna(te[fc], fv);  yte = te["tmax_actual"]

        sc  = StandardScaler()
        Xtrs = sc.fit_transform(Xtr);  Xtes = sc.transform(Xte)

        lin = Ridge(alpha=1.0); lin.fit(Xtrs, ytr)
        mae_l = mean_absolute_error(yte, lin.predict(Xtes)); ml_l.append(mae_l)

        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42)
        gbm.fit(Xtr, ytr)
        mae_g = mean_absolute_error(yte, gbm.predict(Xte)); mg.append(mae_g)

        print(f"  {fold+1:<6} {len(tr):<8} {len(te):<8} {mae_l:>7.2f}C  {mae_g:>7.2f}C")

    if ml_l:
        print(f"\n  Avg Ridge: {np.mean(ml_l):.2f}C  |  Avg GBM: {np.mean(mg):.2f}C")
    return {"ridge_mae": float(np.mean(ml_l)) if ml_l else None,
            "gbm_mae":   float(np.mean(mg))   if mg   else None}


# =============================================================================
# STAGE 5d: FINAL MODEL
# =============================================================================

def train_final(df: pd.DataFrame):
    fc = feature_cols(df)
    m, sc, us, fv = fit_model(df[fc], df["tmax_actual"])
    kind = "GBM" if not us else "Ridge"
    print(f"  [TRAIN] {kind} on {len(df)} days  |  {len(fc)} features")

    imp = pd.Series(
        m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_),
        index=fc).sort_values(ascending=False)
    print(f"\n  Top 8 features:")
    mi = imp.max()
    for f_, v in imp.head(8).items():
        bar = "X" * max(1, int(v * 28 / mi))
        print(f"    {f_:<44} {bar} {v:.4f}")

    return m, sc, us, fv, fc


# =============================================================================
# STAGE 6: FORECAST
# =============================================================================

def make_forecast(model, scaler, use_scaling, fill_values, fc,
                  df, corrector, error_series):
    latest = df.iloc[-1].copy()
    latest["tmax_yesterday"]  = df["tmax_actual"].iloc[-1]
    latest["tmax_2days_ago"]  = df["tmax_actual"].iloc[-2]
    latest["temp_trend_3day"] = df["tmax_actual"].iloc[-3:].mean()

    ml_p         = predict_row(model, scaler, use_scaling, fill_values, fc, latest)
    last_observed = float(df["tmax_actual"].iloc[-1])

    # Pass ml_p and last_observed so the corrector can apply jump dampening.
    # When the model predicts a large departure from recent observed temps,
    # the recent error history is from a different regime and gets down-weighted.
    rt    = corrector.correction(model_forecast=ml_p, last_observed=last_observed)
    final = round(ml_p + rt, 1)

    es    = float(error_series.std())
    eb    = float(error_series.mean())

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


# =============================================================================
# STAGE 7: BETS
# =============================================================================

def bucket_probs(forecast, error_std, buckets):
    from scipy.stats import norm
    probs = {}
    for i, b in enumerate(buckets):
        if   b == buckets[0]:  p = norm.cdf((buckets[i+1]+b)/2, forecast, error_std)
        elif b == buckets[-1]: p = 1 - norm.cdf((b+buckets[i-1])/2, forecast, error_std)
        else:
            lo = (b+buckets[i-1])/2; hi = (buckets[i+1]+b)/2
            p  = norm.cdf(hi, forecast, error_std) - norm.cdf(lo, forecast, error_std)
        probs[b] = round(float(p), 4)
    return probs


def bet_recs(ml_probs, market_prices, min_edge=0.05):
    recs = []
    for b, mp in ml_probs.items():
        if b not in market_prices: continue
        mkt  = market_prices[b]; edge = mp - mkt
        if abs(edge) >= min_edge:
            kelly = edge/(1-mkt) if edge>0 else edge/mkt
            recs.append({"bucket": b, "ml_prob": f"{mp:.1%}",
                         "market_price": f"{mkt:.1%}", "edge": f"{edge:+.1%}",
                         "kelly": f"{kelly:.3f}",
                         "action": "BET YES" if edge>0 else "BET NO",
                         "confidence": "HIGH" if abs(edge)>0.10 else "MEDIUM"})
    recs.sort(key=lambda x: abs(float(x["edge"].replace("%",""))/100), reverse=True)
    return recs


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    station:            str,
    city:               str,
    timezone:           str,
    data_folder:        str   = "mos_data",
    buckets:            list  = None,
    market_prices:      dict  = None,
    temp_min:           float = -40.0,
    temp_max:           float = 60.0,
    initial_train_days: int   = 365,
    run_walk_forward:   bool  = False,
    corrector_seed_days: int  = 14,
    corrector_decay:     float = 0.7,
    corrector_jump_scale: float = 2.0,
):
    """
    Parameters
    ----------
    station              : ICAO code  e.g. "VILK", "LEMD"
    city                 : Must match 'city' column in historical CSV
    timezone             : IANA string  e.g. "Asia/Kolkata", "Europe/Madrid"
    temp_min/max         : Physical bounds for cleaning (defaults work globally)
    initial_train_days   : Min days before first walk-forward prediction
    run_walk_forward     : Full multi-year walk-forward (slow). False = just seed.
    corrector_seed_days  : How many recent days to use for seeding the corrector.
                           14 is fast (~5s). Increase to 30 for more stable seeding.
    """
    if buckets is None:
        buckets = [38, 39, 40, 41, 42, 43, 44]

    try: pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(f"Unknown timezone '{timezone}'")

    print("=" * 65)
    print(f"  MOS  |  {station}  |  {city}  |  {timezone}")
    print("=" * 65)

    print("\n[1] LOAD")
    rm = load_metar(station, data_folder)
    mo = load_model_forecasts(city, data_folder)
    print(f"  METAR: {len(rm):,} rows  |  Model: {len(mo):,} rows")

    print("\n[2] CLEAN")
    rm = clean_metar(rm, temp_min, temp_max)
    mo = clean_model(mo, temp_min, temp_max)

    print("\n[3] ALIGN")
    md = daily_metar(rm, station, timezone)
    od = daily_model(mo, timezone)
    pd_ = pair(md, od)

    print("\n[4] FEATURES")
    fd = build_features(pd_)
    fc = feature_cols(fd)

    # Walk-forward (full, optional)
    wf = pd.DataFrame()
    if run_walk_forward:
        print("\n[5a] WALK-FORWARD VALIDATION")
        wf = walk_forward_validate(fd, initial_train_days)

    # Quick CV diagnostic
    print("\n[5b] CROSS-VALIDATION (diagnostic)")
    cv = cross_validate(fd)

    # Final model on full data
    print("\n[5c] FINAL MODEL")
    model, scaler, use_sc, fill_v, fc = train_final(fd)

    # Seed corrector from real out-of-sample errors
    # This ALWAYS runs — it is fast and ensures the corrector has real signal.
    print(f"\n[5d] SEEDING REAL-TIME CORRECTOR  (last {corrector_seed_days} days OOS)")
    corrector = seed_corrector(fd, seed_days=corrector_seed_days,
                              decay=corrector_decay, jump_scale=corrector_jump_scale)

    # Error series for CI width
    # If walk-forward ran, use those honest errors. Otherwise use in-sample.
    if len(wf) > 0:
        err_series = wf["err_final"]
        print(f"\n  CI width from walk-forward errors (honest): std={err_series.std():.2f}C")
    else:
        Xall  = safe_fillna(fd[fc], fill_v)
        pall  = model.predict(scaler.transform(Xall) if use_sc else Xall)
        err_series = pd.Series(fd["tmax_actual"].values - pall)

    print("\n[6] FORECAST FOR TOMORROW")
    pred = make_forecast(model, scaler, use_sc, fill_v, fc, fd, corrector, err_series)

    print(f"\n  ML model forecast    : {pred['ml_forecast']}C")
    print(f"  Last observed tmax   : {pred['last_observed_tmax']}C")
    print(f"  Forecast jump        : {pred['forecast_jump']:+.1f}C")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Corrector detail     : {pred['corrector_summary']}")
    print(f"  Real-time correction : {pred['realtime_correction']:+.2f}C")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Final forecast       : {pred['final_forecast']}C")
    print(f"  Error std (1 sigma)  : +/-{pred['error_std']}C")
    print(f"  80% CI               : {pred['ci_80'][0]}C – {pred['ci_80'][1]}C")
    print(f"  95% CI               : {pred['ci_95'][0]}C – {pred['ci_95'][1]}C")

    print("\n[7] BET RECOMMENDATION")
    probs = bucket_probs(pred["final_forecast"], pred["error_std"], buckets)

    if market_prices is None:
        raw = {b: np.random.uniform(0.05, 0.35) for b in buckets}
        tot = sum(raw.values())
        market_prices = {b: round(v/tot, 3) for b, v in raw.items()}
        print("  (Simulated market prices)")

    print(f"\n  {'Bucket':<9} {'ML%':>7}  {'Market%':>8}  {'Edge':>7}  Action")
    print("  " + "-" * 48)
    for b in buckets:
        mp   = probs.get(b, 0); mkt = market_prices.get(b, 0); e = mp - mkt
        flag = "  BET" if abs(e) >= 0.05 else ""
        print(f"  {b}C      {mp:>6.1%}   {mkt:>6.1%}   {e:>+6.1%}{flag}")

    bets = bet_recs(probs, market_prices)
    if bets:
        print("\n  Recommended bets:")
        for bt in bets:
            print(f"    {bt['action']} {bt['bucket']}C  "
                  f"edge={bt['edge']}  kelly={bt['kelly']}  {bt['confidence']}")
    else:
        print("\n  No bets above edge threshold.")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)

    return {"model": model, "scaler": scaler, "use_scaling": use_sc,
            "fill_values": fill_v, "feature_cols": fc,
            "corrector": corrector, "wf_results": wf, "cv_results": cv,
            "forecast": pred, "paired_df": pd_, "featured_df": fd, "bets": bets}


# =============================================================================
# ENTRY POINTS
# =============================================================================

if __name__ == "__main__":

    # Lucknow
#     cities = [
#     {"name": "beijing", "station": "ZBAA", "timezone": "Asia/Shanghai"},
#     {"name": "london", "station": "EGLC", "timezone": "Europe/London"},
#     {"name": "tokyo", "station": "RJTT", "timezone": "Asia/Tokyo"},
#     {"name": "lucknow", "station": "VILK", "timezone": "Asia/Kolkata"},
#     {"name": "mexico-city", "station": "MMMX", "timezone": "America/Mexico_City"},
#     {"name": "nyc", "station": "LGA", "timezone": "America/New_York"},
#     {"name": "toronto", "station": "CYYZ", "timezone": "America/Toronto"},
#     {"name": "chicago", "station": "ORD", "timezone": "America/Chicago"},
#     {"name": "atlanta", "station": "ATL", "timezone": "America/New_York"},
#     {"name": "dallas", "station": "DAL", "timezone": "America/Chicago"},
#     {"name": "denver", "station": "BKF", "timezone": "America/Denver"},
#     {"name": "san-francisco", "station": "SFO", "timezone": "America/Los_Angeles"},
#     {"name": "houston", "station": "HOU", "timezone": "America/Chicago"},
#     {"name": "miami", "station": "MIA", "timezone": "America/New_York"},
#     {"name": "los-angeles", "station": "LAX", "timezone": "America/Los_Angeles"},
#     {"name": "austin", "station": "AUS", "timezone": "America/Chicago"},
#     {"name": "seattle", "station": "SEA", "timezone": "America/Los_Angeles"},
#     {"name": "panama-city", "station": "MPMG", "timezone": "America/Panama"},
#     {"name": "sao-paulo", "station": "SBGR", "timezone": "America/Sao_Paulo"},
#     {"name": "buenos-aires", "station": "SAEZ", "timezone": "America/Argentina/Buenos_Aires"},
#     {"name": "wellington", "station": "NZWN", "timezone": "Pacific/Auckland"},
#     {"name": "jakarta", "station": "WIHH", "timezone": "Asia/Jakarta"},
#     {"name": "seoul", "station": "RKSI", "timezone": "Asia/Seoul"},
#     {"name": "singapore", "station": "WSSS", "timezone": "Asia/Singapore"},
#     {"name": "hong-kong", "station": "VHHH", "timezone": "Asia/Hong_Kong"},
#     {"name": "shanghai", "station": "ZSPD", "timezone": "Asia/Shanghai"},
#     {"name": "taipei", "station": "RCSS", "timezone": "Asia/Taipei"},
#     {"name": "kuala-lumpur", "station": "WMKK", "timezone": "Asia/Kuala_Lumpur"},
#     {"name": "chongqing", "station": "ZUCK", "timezone": "Asia/Shanghai"},
#     {"name": "chengdu", "station": "ZUUU", "timezone": "Asia/Shanghai"},
#     {"name": "busan", "station": "RKPK", "timezone": "Asia/Seoul"},
#     {"name": "cape-town", "station": "FACT", "timezone": "Africa/Johannesburg"},
#     {"name": "lagos", "station": "DNMM", "timezone": "Africa/Lagos"},
#     {"name": "jeddah", "station": "OEJN", "timezone": "Asia/Riyadh"},
#     {"name": "tel-aviv", "station": "LLBG", "timezone": "Asia/Jerusalem"},
#     {"name": "munich", "station": "EDDM", "timezone": "Europe/Berlin"},
#     {"name": "paris", "station": "LFPB", "timezone": "Europe/Paris"},
#     {"name": "ankara", "station": "LTAC", "timezone": "Europe/Istanbul"},
#     {"name": "istanbul", "station": "LTFM", "timezone": "Europe/Istanbul"},
#     {"name": "moscow", "station": "UUEE", "timezone": "Europe/Moscow"},
#     {"name": "madrid", "station": "LEMD", "timezone": "Europe/Madrid"},
#     {"name": "helsinki", "station": "EFHK", "timezone": "Europe/Helsinki"},
#     {"name": "amsterdam", "station": "EHAM", "timezone": "Europe/Amsterdam"},
#     {"name": "warsaw", "station": "EPWA", "timezone": "Europe/Warsaw"},
#     {"name": "milan", "station": "LIMC", "timezone": "Europe/Rome"}
# ]
    all_city_names = [c["name"] for c in cities]

    while True:
        city_name = input("Enter city name: ").strip().lower()
        selected_city = next((c for c in cities if c["name"] == city_name), None)
        if city_name in all_city_names:
            print(f"{city_name} found!")
            break
        else:
            print("City not found. Please try again.")

    result = run_pipeline(
        station             = selected_city["station"],
        city                = selected_city["name"],
        timezone            = selected_city["timezone"],
        data_folder         = "mos_data",
        buckets             = [20,21,22,23,24,25,26,27,28,29,30,31],
        market_prices       = None,
        initial_train_days  = 1400,
        run_walk_forward    = False,   # set True for full diagnostic (slow)
        corrector_seed_days = 14,      # increase to 30 for more stable seeding
    )

    # Madrid
    # result = run_pipeline(
    #     station             = "LEMD",
    #     city                = "Madrid",
    #     timezone            = "Europe/Madrid",
    #     data_folder         = "mos_data",
    #     buckets             = [14,15,16,17,18,19,20,21,22,23,24,25,26],
    #     market_prices       = None,
    #     initial_train_days  = 365,
    #     run_walk_forward    = False,
    #     corrector_seed_days = 14,
    # )

