import datetime
import re
import ast
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
try:
    import pytz
except ImportError:
    raise ImportError("pip install pytz")

from config import MODELS, SKY_MAP, cities, is_fahrenheit_city, c_to_f
from features import _apparent_temp_c
warnings.filterwarnings("ignore")


def load_metar(station: str, data_folder: str = "Data") -> pd.DataFrame:
    data_folder = Path(data_folder)
    file_path = data_folder / "metar_data"
    df = pd.read_csv(Path(file_path) / f"{station}.csv")
    df["valid"] = pd.to_datetime(df["valid"], utc=False)
    if "sknt" in df.columns and "wind_kt" not in df.columns:
        df = df.rename(columns={"sknt": "wind_kt"})
    return df

def load_model_forecasts(city: str, data_folder: str = "Data") -> pd.DataFrame:
    data_folder = Path(data_folder)
    file_path = data_folder / "forecast_data"

    p1 = Path(file_path) / f"historical_{city}.csv"
    p2 = Path(file_path) / "historical.csv"
    path = p1 if p1.exists() else (p2 if p2.exists() else None)
    if path is None:
        raise FileNotFoundError(f"No model data found. Tried {p1} and {p2}")

    df = pd.read_csv(path)
    df = df[df["city"].str.lower() == city.lower()].copy()
    if len(df) == 0:
        raise ValueError(f"No rows for city='{city}' in {path}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    # ── Cloud cover ───────────────────────────────────────────────────────────
    # Open-Meteo now provides a single total cloud_cover column per model.
    # If the old three-layer columns (low/mid/high) are present, sum them;
    # otherwise use cloud_cover_{m} directly.  Either way the result lives in
    # cloud_cover_{m} so all downstream code is unchanged.
    for m in MODELS:
        if f"cloud_cover_{m}" in df.columns:
            # New format: total cloud cover already available — clip to [0,100]
            df[f"cloud_cover_{m}"] = pd.to_numeric(
                df[f"cloud_cover_{m}"], errors="coerce").clip(upper=100)
        else:
            # Legacy format: sum of three layers
            lo = pd.to_numeric(df.get(f"cloud_cover_low_{m}",  0), errors="coerce").fillna(0)
            mi = pd.to_numeric(df.get(f"cloud_cover_mid_{m}",  0), errors="coerce").fillna(0)
            hi = pd.to_numeric(df.get(f"cloud_cover_high_{m}", 0), errors="coerce").fillna(0)
            df[f"cloud_cover_{m}"] = (lo + mi + hi).clip(upper=100)

    # ── Standard ensemble means ───────────────────────────────────────────────
    for prefix, out in [("temperature_2m",       "ens_temp"),
                         ("relative_humidity_2m", "ens_humidity"),
                         ("shortwave_radiation",  "ens_solar"),
                         ("wind_speed_10m",       "ens_wind"),
                         ("dew_point_2m",         "ens_dewpoint")]:
        cols = [f"{prefix}_{m}" for m in MODELS if f"{prefix}_{m}" in df.columns]
        if cols:
            df[out] = df[cols].mean(axis=1)

    cc = [f"cloud_cover_{m}" for m in MODELS if f"cloud_cover_{m}" in df.columns]
    if cc:
        df["ens_cloud"] = df[cc].mean(axis=1)

    # ── Wind direction: vector-average across models ───────────────────────────
    # Raw degree averaging is wrong (359° and 1° → 180°).  Convert each model's
    # direction to unit-circle U/V components, average, store as ens_wind_u/v.
    wind_dir_cols = [f"wind_direction_10m_{m}" for m in MODELS
                     if f"wind_direction_10m_{m}" in df.columns]
    if wind_dir_cols:
        u_parts, v_parts = [], []
        for col in wind_dir_cols:
            rad = np.radians(pd.to_numeric(df[col], errors="coerce"))
            u_parts.append(-np.sin(rad))   # meteorological convention: wind FROM
            v_parts.append(-np.cos(rad))
        df["ens_wind_u"] = pd.concat(u_parts, axis=1).mean(axis=1)
        df["ens_wind_v"] = pd.concat(v_parts, axis=1).mean(axis=1)

    return df.sort_values("date").reset_index(drop=True)


# Cleaning data

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



def daily_metar(metar_df, station, tz):
    df = metar_df[metar_df["station"] == station].copy()
    df["lt"] = to_local(df["valid"], tz)
    df["ld"] = df["lt"].dt.date
    last_date = df["ld"].max()
    rows = []
    for d, g in df.groupby("ld"):
        # Require >= 20 observations for a complete day.
        # Exception: the most recent local date may be partial (today still in
        # progress). Keep it regardless of obs count so its tmax_actual feeds
        # the corrector seeder. Without this, cities like Wellington (UTC+12)
        # always lag one day behind because today's local date never accumulates
        # 20 obs before the pipeline runs.
        # is_last = (d == last_date)
        if len(g) < 20 : # if len(g) < 20 and not is_last:
            continue
        idx      = g["tmpc"].idxmax()
        peak     = g[g["lt"].dt.hour.between(11, 15)]
        if len(peak) == 0: peak = g
        cloud    = peak["skyc1"].map(SKY_MAP).fillna(50).mean()

        t_peak   = float(peak["tmpc"].mean())
        rh_peak  = float(peak["relh"].mean())
        wnd_peak = float(peak["wind_kt"].mean())

        # Dewpoint at peak — use METAR dwpc column when available
        dwp_peak = np.nan
        if "dwpc" in peak.columns:
            dwp_val = pd.to_numeric(peak["dwpc"], errors="coerce")
            if dwp_val.notna().any():
                dwp_peak = float(dwp_val.mean())
        # Fallback: derive dewpoint from temperature and RH using August-Roche-Magnus
        # Td ≈ T − (100 − RH) / 5  (accurate to ~1°C for RH > 50%)
        if np.isnan(dwp_peak):
            dwp_peak = t_peak - (100.0 - rh_peak) / 5.0

        at_peak = _apparent_temp_c(t_peak, rh_peak, wnd_peak)

        rows.append({
            "date":               pd.Timestamp(d),
            "tmax_actual":        round(float(g.loc[idx, "tmpc"]), 1),
            "tmax_hour_local":    int(g.loc[idx, "lt"].hour),
            "wind_at_peak":       round(wnd_peak, 1),
            "humidity_at_peak":   round(rh_peak, 1),
            "cloud_at_peak":      round(float(cloud), 1),
            "dewpoint_at_peak":   round(dwp_peak, 2),
            "apparent_temp_peak": round(at_peak, 2),
            "obs_count":          len(g),
            "is_partial":         len(g) < 20,
        })
    r = pd.DataFrame(rows)
    partial = int(r["is_partial"].sum()) if "is_partial" in r.columns else 0
    print(f"  [ALIGN] {len(r)} daily METAR records ({station})"
          + ("  [1 partial day kept]" if partial else ""))
    return r


def daily_model(model_df, tz):
    df    = model_df.copy()
    # df["date"] may be tz-naive (load_model_forecasts strips tz) or tz-aware.
    # Always localise as UTC before converting to local time.
    if df["date"].dt.tz is None:
        # load_model_forecasts strips tz with tz_localize(None), leaving the
        # column tz-naive but numerically UTC. Re-localise here before converting.
        dt_utc = df["date"].dt.tz_localize("UTC")
    else:
        dt_utc = df["date"].dt.tz_convert("UTC")
    df["lt"] = to_local(dt_utc, tz)
    df["ld"] = df["lt"].dt.date
    tcols = [f"temperature_2m_{m}" for m in MODELS if f"temperature_2m_{m}" in df.columns]
    rows  = []
    for d, g in df.groupby("ld"):
        # Strict filter: only complete days (>= 20 obs) go into the paired
        # training data. Partial/future dates are handled separately by
        # get_tomorrow_model_row() which reads raw hourly data directly.
        if len(g) < 20:
            continue
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
                         ("ens_solar",    "solar_ghi_forecast"),
                         ("ens_dewpoint", "dewpoint_forecast"),
                         ("ens_wind_u",   "wind_u_forecast"),
                         ("ens_wind_v",   "wind_v_forecast")]:
            if src in df.columns:
                row[dst] = float(pk[src].mean())
        rows.append(row)
    r = pd.DataFrame(rows)
    print(f"  [ALIGN] {len(r)} daily model records")
    return r


def pair(metar_d, model_d):
    """
        Exclude partial METAR days from training.
        daily_metar keeps the most recent partial day so its tmax_actual can
        feed the corrector seeder, but a partial day's tmax is unreliable for
        training — the day is not over yet. If we leave it in, fd ends one day
        later than it should, pushing the target date forward by one.
    """
    if "is_partial" in metar_d.columns:
        train_metar = metar_d[~metar_d["is_partial"]].copy()
    else:
        train_metar = metar_d
    p = pd.merge(train_metar, model_d, on="date", how="inner")
    print(f"  [PAIR]  {len(p)} paired days")
    if len(p) == 0:
        raise ValueError("No overlapping dates — check timezone / city name / date ranges")
    return p


def get_tomorrow_model_row(mo_raw: pd.DataFrame, tz: str,
                           target_date: str) -> pd.Series:
    """
    Aggregate the raw hourly model data for target_date into a single daily row,
    using the same aggregation logic as daily_model but with NO obs-count filter.

    This is used exclusively by make_forecast to get tomorrow's NWP features.
    It must bypass the >= 20 obs filter because:
      - Tomorrow's data may only be a partial day in the download window
      - Even a few hours of forecast data are enough for daily max temperature

    Parameters
    ----------
    mo_raw      : raw model DataFrame from load_model_forecasts (tz-naive UTC)
    tz          : IANA timezone string
    target_date : "YYYY-MM-DD" local date being predicted

    Returns
    -------
    pd.Series with the same columns as daily_model rows, or raises ValueError
    if no model data exists for target_date.
    """
    df = mo_raw.copy()
    if df["date"].dt.tz is None:
        dt_utc = df["date"].dt.tz_localize("UTC")
    else:
        dt_utc = df["date"].dt.tz_convert("UTC")
    df["lt"] = to_local(dt_utc, tz)
    df["ld"] = df["lt"].dt.date

    target_d = pd.Timestamp(target_date).date()
    g = df[df["ld"] == target_d]

    if len(g) == 0:
        local_dates = sorted(df["ld"].unique())
        raise ValueError(
            f"[FORECAST] No model data found for {target_date} (local). "
            f"Available local dates: {local_dates[0]} to {local_dates[-1]}. "
            f"Ensure your model CSV includes a forecast for {target_date}."
        )

    tcols = [f"temperature_2m_{m}" for m in MODELS if f"temperature_2m_{m}" in df.columns]
    row = {"date": pd.Timestamp(target_date)}
    for c in tcols:
        row[f"{c}_max"] = float(g[c].max())
    mx = [row[f"{c}_max"] for c in tcols]
    row["model_spread"]   = float(max(mx) - min(mx))
    row["model_mean_max"] = float(np.mean(mx))
    row["model_std_max"]  = float(np.std(mx))
    pk = g[g["lt"].dt.hour.between(11, 15)]
    if len(pk) == 0:
        pk = g
    for src_col, dst_col in [("ens_cloud",    "cloud_cover_forecast"),
                              ("ens_humidity", "humidity_forecast"),
                              ("ens_wind",     "wind_forecast"),
                              ("ens_solar",    "solar_ghi_forecast"),
                              ("ens_dewpoint", "dewpoint_forecast"),
                              ("ens_wind_u",   "wind_u_forecast"),
                              ("ens_wind_v",   "wind_v_forecast")]:
        if src_col in df.columns:
            row[dst_col] = float(pk[src_col].mean())

    print(f"  [FORECAST] Tomorrow model row ({target_date}): "
          f"{len(g)} hourly obs used, "
          f"mean_max={row.get('model_mean_max', float('nan')):.1f}C")
    return pd.Series(row)

def to_local(series: pd.Series, tz_name: str) -> pd.Series:
    """DST-correct UTC -> local. Never use a fixed offset for DST cities."""
    tz = pytz.timezone(tz_name)
    s  = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)