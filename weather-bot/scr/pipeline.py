import sys
import datetime
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

from corrector import load_corrector, save_corrector
from data import (load_metar, load_model_forecasts, clean_metar, clean_model, daily_metar, daily_model, pair)
from features import build_features, feature_cols, safe_fillna
from model import (walk_forward_validate, cross_validate, train_final, seed_corrector, predict_row)
from forecast import (make_forecast, get_target_date)
from config import cities
import sys
try:
    import pytz
except ImportError:
    raise ImportError("pip install pytz")

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent.parent

def run_pipeline(
    station:             str,
    city:                str,
    timezone:            str,
    data_folder:         str   = "Data",
    temp_min:            float = -40.0,
    temp_max:            float = 60.0,
    initial_train_days:  int   = 365,
    run_walk_forward:    bool  = False,
    corrector_seed_days: int   = 14,
    corrector_decay:     float = 0.7,
    corrector_jump_scale: float = 2.0,
):
    """
    Parameters
    ----------
    station              : ICAO code  e.g. "VILK", "LEMD"
    city                 : Must match 'city' column in historical CSV
    timezone             : IANA string  e.g. "Asia/Kolkata", "Europe/Madrid"
    data_folder          : Root folder for all data (METAR, model, market CSVs)
    temp_min/max         : Physical bounds for cleaning (defaults work globally)
    initial_train_days   : Min days before first walk-forward prediction
    run_walk_forward     : Full multi-year walk-forward (slow). False = just seed.
    corrector_seed_days  : How many recent days to use for seeding the corrector.
        """

    import traceback, sys
    _orig = sys.stdout
    try:
        pytz.timezone(timezone)
    except Exception as e:
        raise ValueError(f"Unknown timezone '{timezone}'")

    log_dir = Path(data_folder) / "logs" / city
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRASH at log_dir.mkdir: {e}")
        traceback.print_exc()
        return





    import sys
    log_dir = Path(data_folder) / "logs" / city
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts   = datetime.datetime.now().strftime("%Y-%m-%d_TIme_%H-%M")
    log_path = log_dir / f"{city}_{run_ts}.log"

    class _Tee:
        """Writes to both the original stdout and a log file simultaneously."""
        def __init__(self, stream, filepath):
            self._s = stream
            self._f = open(filepath, "w", encoding="utf-8", buffering=1)
        def write(self, data):
            self._s.write(data)
            self._f.write(data)
        def flush(self):
            self._s.flush()
            self._f.flush()
        def close(self):
            self._f.close()
        # Delegate everything else (isatty, fileno etc.) to original stream
        def __getattr__(self, name):
            return getattr(self._s, name)

    _orig_stdout = sys.stdout
    sys.stdout   = _Tee(sys.stdout, log_path)
    print(f"  [LOG] Writing to {log_path}")

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

    
    print(f"\n[5d] CORRECTOR")
    corrector = load_corrector(city, data_folder,
                               decay=corrector_decay,
                               jump_scale=corrector_jump_scale)
    if corrector is None:
        
        corrector = seed_corrector(fd, seed_days=corrector_seed_days,
                                   decay=corrector_decay,
                                   jump_scale=corrector_jump_scale)
    else:
        print(f"  [CORRECTOR] State: {corrector.summary()}")

    # Error series for CI width
    # If walk-forward ran, use those honest errors. Otherwise use in-sample.
    if len(wf) > 0:
        err_series = wf["err_final"]
        print(f"\n  CI width from walk-forward errors (honest): std={err_series.std():.2f}C")
    else:
        Xall  = safe_fillna(fd[fc], fill_v)
        pall  = model.predict(scaler.transform(Xall) if use_sc else Xall)
        err_series = pd.Series(fd["tmax_actual"].values - pall)

    target_date = get_target_date(fd)

    # Fall back to simulation only when the market file is genuinely absent

    last_completed_row = fd.iloc[-1].copy()
    latest_ml_pred = predict_row(model, scaler, use_sc, fill_v, fc, last_completed_row)
    latest_actual = float(last_completed_row["tmax_actual"])
    latest_date_str = last_completed_row["date"].strftime("%Y-%m-%d")
    
    corrector.update(latest_date_str, latest_ml_pred, latest_actual)
    print(f"\n  [CORRECTOR] Logged live error ({latest_date_str}): Pred {latest_ml_pred:.1f}C vs Actual {latest_actual:.1f}C -> Error {latest_actual - latest_ml_pred:+.1f}C")
    # -----------------------------------------------------------------------

    print(f"\n[6] FORECAST FOR {target_date}")
    pred = make_forecast(model, scaler, use_sc, fill_v, fc,
                         fd, mo, timezone, target_date, corrector, err_series)

    print(f"\n  Target date          : {target_date}")
    print(f"  ML model forecast    : {pred['ml_forecast']}C")
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


    print(f"today's actual tmax for {target_date} is observed.")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)
    
    # ── Close log file ────────────────────────────────────────────────
    sys.stdout.close()
    sys.stdout = _orig_stdout
    print(f"  [LOG] Run log saved -> {log_path}")

    return {"model": model, "scaler": scaler, "use_scaling": use_sc,
            "fill_values": fill_v, "feature_cols": fc,
            "corrector": corrector, "wf_results": wf, "cv_results": cv,
            "forecast": pred, "paired_df": pd_, "featured_df": fd,
            "target_date": target_date}


