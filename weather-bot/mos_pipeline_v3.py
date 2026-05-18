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
import mos_pipeline_v4
from datetime import date
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
from variable import cities
import sys
try:
    import pytz
except ImportError:
    raise ImportError("pip install pytz")

warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

cities = [
    {"name": "beijing", "station": "ZBAA", "timezone": "Asia/Shanghai", "lat": 40.0801, "lon": 116.5846, "network": "CN__ASOS"},
    {"name": "london", "station": "EGLC", "timezone": "Europe/London", "lat": 51.5085, "lon": -0.1257, "network": "GB__ASOS"},
    {"name": "tokyo", "station": "RJTT", "timezone": "Asia/Tokyo", "lat": 35.6895, "lon": 139.6917, "network": "JP__ASOS"},
    {"name": "lucknow", "station": "VILK", "timezone": "Asia/Kolkata", "lat": 26.74, "lon": 80.86, "network": "IN__ASOS"},
    {"name": "mexico-city", "station": "MMMX", "timezone": "America/Mexico_City", "lat": 19.44, "lon": -99.08, "network": "MX__ASOS"},
    {"name": "nyc", "station": "LGA", "timezone": "America/New_York", "lat": 40.76, "lon": -73.86, "network": "NY_ASOS"},
    {"name": "toronto", "station": "CYYZ", "timezone": "America/Toronto", "lat": 43.71, "lon": -79.66, "network": "CA_ON_ASOS"},
    {"name": "chicago", "station": "ORD", "timezone": "America/Chicago", "lat": 41.98, "lon": -87.91, "network": "IL_ASOS"},
    {"name": "atlanta", "station": "ATL", "timezone": "America/New_York", "lat": 33.64, "lon": -84.41, "network": "GA_ASOS"},
    {"name": "dallas", "station": "DAL", "timezone": "America/Chicago", "lat": 32.85, "lon": -96.87, "network": "TX_ASOS"},
    {"name": "denver", "station": "BKF", "timezone": "America/Denver", "lat": 39.7, "lon": -104.76, "network": "CO_ASOS"},
    {"name": "san-francisco", "station": "SFO", "timezone": "America/Los_Angeles", "lat": 37.62, "lon": -122.39, "network": "CA_ASOS"},
    {"name": "houston", "station": "HOU", "timezone": "America/Chicago", "lat": 29.63, "lon": -95.25, "network": "TX_ASOS"},
    {"name": "miami", "station": "MIA", "timezone": "America/New_York", "lat": 25.85, "lon": -80.24, "network": "FL_ASOS"},
    {"name": "los-angeles", "station": "LAX", "timezone": "America/Los_Angeles", "lat": 33.96, "lon": -118.4, "network": "CA_ASOS"},
    {"name": "austin", "station": "AUS", "timezone": "America/Chicago", "lat": 30.16, "lon": -97.69, "network": "TX_ASOS"},
    {"name": "seattle", "station": "SEA", "timezone": "America/Los_Angeles", "lat": 47.44, "lon": -122.3, "network": "WA_ASOS"},
    {"name": "panama-city", "station": "MPMG", "timezone": "America/Panama", "lat": 8.98, "lon": 79.56, "network": "PA__ASOS"},
    {"name": "sao-paulo", "station": "SBGR", "timezone": "America/Sao_Paulo", "lat": -23.42, "lon": -46.48, "network": "BR__ASOS"},
    {"name": "buenos-aires", "station": "SAEZ", "timezone": "America/Argentina/Buenos_Aires", "lat": -34.79, "lon": -58.52, "network": "AR__ASOS"},
    {"name": "wellington", "station": "NZWN", "timezone": "Pacific/Auckland", "lat": -41.32, "lon": 174.8, "network": "NF__ASOS"},
    {"name": "jakarta", "station": "WIHH", "timezone": "Asia/Jakarta", "lat": -6.26, "lon": 106.89, "network": "ID__ASOS"},
    {"name": "seoul", "station": "RKSI", "timezone": "Asia/Seoul", "lat": 37.49, "lon": 126.49, "network": "KR__ASOS"},
    {"name": "singapore", "station": "WSSS", "timezone": "Asia/Singapore", "lat": 1.35, "lon": 104, "network": "SG__ASOS"},
    {"name": "hong-kong", "station": "VHHH", "timezone": "Asia/Hong_Kong", "lat": 22.2783, "lon": 114.1747, "network": "HK__ASOS"},
    {"name": "shanghai", "station": "ZSPD", "timezone": "Asia/Shanghai", "lat": 31.15, "lon": 121.8, "network": "CN__ASOS"},
    {"name": "taipei", "station": "RCSS", "timezone": "Asia/Taipei", "lat": 25.06, "lon": 121.55, "network": "TW__ASOS"},
    {"name": "kuala-lumpur", "station": "WMKK", "timezone": "Asia/Kuala_Lumpur", "lat": 2.77, "lon": 101.7, "network": "MY__ASOS"},
    {"name": "chongqing", "station": "ZUCK", "timezone": "Asia/Shanghai", "lat": 29.72, "lon": 106.63, "network": "CN__ASOS"},
    {"name": "chengdu", "station": "ZUUU", "timezone": "Asia/Shanghai", "lat": 30.57, "lon": 103.96, "network": "CN__ASOS"},
    {"name": "busan", "station": "RKPK", "timezone": "Asia/Seoul", "lat": 35.18, "lon": 128.95, "network": "KR__ASOS"},
    {"name": "cape-town", "station": "FACT", "timezone": "Africa/Johannesburg", "lat": -33.97, "lon": 18.59, "network": "ZA__ASOS"},
    {"name": "lagos", "station": "DNMM", "timezone": "Africa/Lagos", "lat": 6.45, "lon": 3.39, "network": "NG__ASOS"},
    {"name": "jeddah", "station": "OEJN", "timezone": "Asia/Riyadh", "lat": 21.58, "lon": 39.16, "network": "SA__ASOS"},
    {"name": "tel-aviv", "station": "LLBG", "timezone": "Asia/Jerusalem", "lat": 32.0809, "lon": 34.7806, "network": "IL__ASOS"},
    {"name": "munich", "station": "EDDM", "timezone": "Europe/Berlin", "lat": 48.35, "lon": 11.79, "network": "DE__ASOS"},
    {"name": "paris", "station": "LFPB", "timezone": "Europe/Paris", "lat": 49.02, "lon": 2.59, "network": "FR__ASOS"},
    {"name": "ankara", "station": "LTAC", "timezone": "Europe/Istanbul", "lat": 40.24, "lon": 33.03, "network": "TR__ASOS"},
    {"name": "istanbul", "station": "LTFM", "timezone": "Europe/Istanbul", "lat": 41.0138, "lon": 28.9497, "network": "TR__ASOS"},
    {"name": "moscow", "station": "UUEE", "timezone": "Europe/Moscow", "lat": 55.7522, "lon": 37.6156, "network": "RU__ASOS"},
    {"name": "madrid", "station": "LEMD", "timezone": "Europe/Madrid", "lat": 40.45, "lon": -3.58, "network": "ES__ASOS"},
    {"name": "helsinki", "station": "EFHK", "timezone": "Europe/Helsinki", "lat": 60.32, "lon": 24.97, "network": "FI__ASOS"},
    {"name": "amsterdam", "station": "EHAM", "timezone": "Europe/Amsterdam", "lat": 52.31, "lon": 4.76, "network": "NL__ASOS"},
    {"name": "warsaw", "station": "EPWA", "timezone": "Europe/Warsaw", "lat": 52.17, "lon": 20.98, "network": "PL__ASOS"},  
    {"name": "milan", "station": "LIMC", "timezone": "Europe/Rome", "lat": 45.63, "lon": 8.7, "network": "IT__ASOS"}
]

MODEL_ID_MAP = {
    30: "ecmwf_ifs",
    2:  "gfs_seamless",
    16: "gem_seamless",
    20: "icon_seamless",    
    4:  "gfs_hrrr",
    82: "ukmo_seamless",
    60: "ecmwf_ifs025"
}

MODELS = [
    "ecmwf_ifs",
    "ecmwf_ifs025",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless",
    "ukmo_seamless",
]

SKY_MAP = {"NSC": 0, "FEW": 20, "SCT": 45, "BKN": 75, "OVC": 100 , "NCD": 0}


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
        """Record error = actual - ml_prediction. Handles duplicates and backtesting."""
        date_str = str(date)
        new_error = float(actual) - float(ml_prediction)
        
        # Convert history to a dict for easy lookup/overwrite
        history_dict = dict(self._history)
        
        # Update or add the error for this specific date
        history_dict[date_str] = new_error
        
        # Re-sort by date string so the window always captures the most recent chronological days
        sorted_dates = sorted(history_dict.keys())
        
        # Rebuild history and apply the sliding window
        self._history = [(d, history_dict[d]) for d in sorted_dates]
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
            # Clamp to [0, 100] — if correction crosses zero due to floating
            # point the raw formula overshoots and produces negative or >100%
            # values which are confusing in the output.
            if base != 0:
                damp_pct = max(0, min(100, 100 * (1 - abs(damped) / abs(base))))
            else:
                damp_pct = 0
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
# CORRECTOR PERSISTENCE
# =============================================================================

def corrector_state_path(city: str, data_folder: str = "mos_data") -> Path:
    """Path to the JSON file that persists the corrector state between runs."""

    script_dir = Path(__file__).resolve().parent
    data_folder = Path(data_folder)
    full_path = data_folder / "corrector-folder-v3"
    return script_dir / full_path / f"corrector_{city.lower()}.json"


def save_corrector(corrector: RealtimeBiasCorrector,
                   city: str, data_folder: str = "mos_data") -> None:
    """
    Persist corrector state to disk after each run.
    Call this AFTER observing the actual tmax and calling corrector.update().
    In practice: call it at the end of run_pipeline so the next run picks up
    the real error from today's forecast.
    """

    import json
    path = corrector_state_path(city, data_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(corrector.state(), f, default=str)
    print(f"  [CORRECTOR] State saved -> {path.name}")


def load_corrector(city: str, data_folder: str = "mos_data",
                   decay: float = 0.7,
                   jump_scale: float = 2.0) -> "RealtimeBiasCorrector | None":
    """
    Load persisted corrector state from disk.
    Returns None if no saved state exists (first ever run).
    """
    import json
    path = corrector_state_path(city, data_folder)
    if not path.exists():
        return None
    with open(path) as f:
        state = json.load(f)
    corr = RealtimeBiasCorrector.from_state(state)
    print(f"  [CORRECTOR] Loaded saved state from {path.name}  "
          f"({corr.n_days()} days of real errors)")
    return None


def update_and_save_corrector(corrector: RealtimeBiasCorrector,
                               forecast_date: str,
                               ml_prediction: float,
                               actual_tmax: float,
                               city: str,
                               data_folder: str = "mos_data") -> None:
    """
    After you observe the real tmax for a forecast day, call this to:
      1. Record the real error into the corrector
      2. Save the updated state to disk

    Parameters
    ----------
    forecast_date  : The date that was being predicted (YYYY-MM-DD)
    ml_prediction  : The ml_forecast value from that day's run_pipeline output
    actual_tmax    : The real observed tmax for that date
    """
    corrector.update(forecast_date, ml_prediction, actual_tmax)
    error = actual_tmax - ml_prediction
    print(f"  [CORRECTOR] Updated with real error for {forecast_date}: "
          f"predicted={ml_prediction:.1f}C  actual={actual_tmax:.1f}C  "
          f"error={error:+.1f}C")
    # save_corrector(corrector, city, data_folder)


# =============================================================================
# TIMEZONE
# =============================================================================

def to_local(series: pd.Series, tz_name: str) -> pd.Series:
    """DST-correct UTC -> local. Never use a fixed offset for DST cities."""
    tz = pytz.timezone(tz_name)
    s  = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)


# American timezone prefixes — markets for these cities use Fahrenheit
_US_TIMEZONES = {
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "America/Honolulu",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "Pacific/Honolulu",
}

def is_fahrenheit_city(timezone: str) -> bool:
    """Return True ONLY for US cities whose market buckets are in Fahrenheit."""
    return timezone in _US_TIMEZONES


def c_to_f(c: float) -> int:
    """Convert Celsius to Fahrenheit and round to nearest integer."""
    return round(c * 9 / 5 + 32)


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


def load_market_data(city: str, timezone: str,
                     target_date: str,
                     data_folder: str = "mos_data") -> tuple:
    """
    Load buckets AND market prices from  data/<date>/market-<date>.csv

    Parameters
    ----------
    city        : city name matching the CSV "city" column (case-insensitive)
    timezone    : IANA timezone — used only for Fahrenheit city detection
    target_date : "YYYY-MM-DD" of the day being predicted. Must be derived from
                  the last paired training date + 1 day, NOT the system clock.
    data_folder : root data folder (resolved relative to this script)

    Returns
    -------
    (buckets, market_prices)
    buckets       : sorted list of ints in CELSIUS
    market_prices : dict {int_celsius_bucket: float_YES_price}
    On any failure returns (None, None).

    Title parsing
    -------------
    "19°C or below" -> 19,  "21°C" -> 21,  "29°C or higher" -> 29
    American cities: CSV titles in °F, converted back to °C internally.

    Path resolution
    ---------------
    Always relative to this script's own directory.
    """
    import re
    import ast

    script_dir   = Path(__file__).resolve().parent
    data_dir     = script_dir /  "data" / target_date
    csv_path     = data_dir / f"market-{target_date}.csv"

    # --- Existence checks with clear error messages ---
    if not data_dir.exists():
        print(f"  [MARKET] Folder not found: {data_dir}  — will simulate prices.")
        return None, None

    if not csv_path.exists():
        print(f"  [MARKET] File not found: {csv_path}  — will simulate prices.")
        return None, None

    df = pd.read_csv(csv_path)

    # Filter to this city (case-insensitive)
    city_df = df[df["city"].str.lower() == city.lower()].copy()
    if len(city_df) == 0:
        print(f"  [MARKET] City '{city}' not found in {csv_path.name}  — will simulate.")
        return None, None

    fahrenheit = is_fahrenheit_city(timezone)
    prices_c   = {}   # all keys stored in Celsius

    for _, row in city_df.iterrows():
        title = str(row["title"])

        # --- Parse YES price (first element of the JSON list) ---
        try:
            price_list = ast.literal_eval(row["prices"])
            yes_price  = float(price_list[0])
        except Exception:
            print(f"  [MARKET] Could not parse prices column for row: {title!r} — skipping.")
            continue

        # --- Parse temperature integer from title ---
        # Optional leading minus handles negative temps (e.g. "-5°C or below")
        nums = re.findall(r"-?\d+", title)
        if not nums:
            print(f"  [MARKET] No temperature integer found in title: {title!r} — skipping.")
            continue
        temp_raw = int(nums[0])

        # --- For American cities: CSV is in °F, convert back to °C ---
        if fahrenheit:
            temp_c = round((temp_raw - 32) * 5 / 9)
        else:
            temp_c = temp_raw

        prices_c[temp_c] = yes_price

    if not prices_c:
        print(f"  [MARKET] No valid rows parsed from {csv_path.name}  — will simulate.")
        return None, None

    # Buckets = sorted Celsius integers (preserves natural ascending order)
    buckets = sorted(prices_c.keys())

    unit = "°F" if fahrenheit else "°C"
    print(f"  [MARKET] Loaded {len(buckets)} buckets for '{city}' "
          f"from {csv_path.name}  (date: {target_date}, market in {unit})")
    print(f"  [MARKET] Buckets (°C): {buckets}")

    return buckets, prices_c


def load_metar(station: str, data_folder: str = "mos_data") -> pd.DataFrame:
    data_folder = Path(data_folder)
    file_path = data_folder / "metar_data"
    df = pd.read_csv(Path(file_path) / f"{station}.csv")
    df["valid"] = pd.to_datetime(df["valid"], utc=False)
    if "sknt" in df.columns and "wind_kt" not in df.columns:
        df = df.rename(columns={"sknt": "wind_kt"})
    return df


def load_model_forecasts(city: str, data_folder: str = "mos_data") -> pd.DataFrame:
    data_folder = Path(data_folder)
    file_path = data_folder / "forcast_data"

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

    for m in MODELS:
        hi = df.get(f"cloud_cover_{m}", pd.Series(0, index=df.index))
        df[f"cloud_cover_{m}"] = (hi).clip(upper=100)

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
        rows.append({
            "date":             pd.Timestamp(d),
            "tmax_actual":      round(float(g.loc[idx, "tmpc"]), 1),
            "tmax_hour_local":  int(g.loc[idx, "lt"].hour),
            "wind_at_peak":     round(float(peak["wind_kt"].mean()), 1),
            "humidity_at_peak": round(float(peak["relh"].mean()), 1),
            "cloud_at_peak":    round(float(cloud), 1),
            "obs_count":        len(g),
            "is_partial":       len(g) < 20, # "is_partial":       is_last and len(g) < 20,
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
                         ("ens_solar",    "solar_ghi_forecast")]:
            if src in df.columns: row[dst] = float(pk[src].mean())
        rows.append(row)
    r = pd.DataFrame(rows)
    print(f"  [ALIGN] {len(r)} daily model records")
    return r


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
                              ("ens_solar",    "solar_ghi_forecast")]:
        if src_col in df.columns:
            row[dst_col] = float(pk[src_col].mean())

    print(f"  [FORECAST] Tomorrow model row ({target_date}): "
          f"{len(g)} hourly obs used, "
          f"mean_max={row.get('model_mean_max', float('nan')):.1f}C")
    return pd.Series(row)


def pair(metar_d, model_d):
    # Exclude partial METAR days from training.
    # daily_metar keeps the most recent partial day so its tmax_actual can
    # feed the corrector seeder, but a partial day's tmax is unreliable for
    # training — the day is not over yet. If we leave it in, fd ends one day
    # later than it should, pushing the target date forward by one.
    if "is_partial" in metar_d.columns:
        train_metar = metar_d[~metar_d["is_partial"]].copy()
    else:
        train_metar = metar_d
    p = pd.merge(train_metar, model_d, on="date", how="inner")
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

    # Stop at n-1, NOT n.
    # The last row of df (index n-1) is the feature row that make_forecast()
    # uses to predict tomorrow. We have never issued a real forecast for it —
    # its "error" would be in-sample noise, not a genuine out-of-sample error.
    # Including it feeds a spurious data point into the corrector that biases
    # every subsequent forecast. It's the exact cause of the sign-flip bug:
    # the corrector was seeing a -0.2 error for the day it was about to predict.
    seed_end = n - 1
    print(f"  [SEED]  Running {seed_end - start_idx} day(s) of out-of-sample prediction "
          f"to seed real-time corrector...")

    for test_idx in range(start_idx, seed_end):
        train = df.iloc[:test_idx]
        row   = df.iloc[test_idx]
        m, sc, us, fv = fit_model(train[fc], train["tmax_actual"])
        ml_p  = predict_row(m, sc, us, fv, fc, row)
        actual = float(row["tmax_actual"])
        error  = actual - ml_p
        date_s = row["date"].strftime("%Y-%m-%d")
        corr.update(row["date"], ml_p, actual)
        print(f"  [CORRECTOR] Logged live error ({date_s}): "
              f"Pred {ml_p:.1f}C  Actual {actual:.1f}C  Error {error:+.1f}C")
        
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
        # Use last observed tmax (previous row's actual) for jump dampening,
        # matching exactly what the production corrector does in make_forecast.
        # Previously called corr.correction() with no args, so dampening was
        # never applied — making walk-forward evaluate a different behaviour
        # than production and producing a slightly optimistic MAE estimate.
        last_obs = float(df["tmax_actual"].iloc[idx - 1]) if idx > 0 else None
        rt     = corr.correction(model_forecast=ml_p, last_observed=last_obs)
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

    # ── A) Lag features (from METAR actuals up to today) ─────────────────────
    # build_features defines these via .shift() on tmax_actual:
    #   tmax_yesterday  = shift(1) → for tomorrow's row = today's actual
    #   tmax_2days_ago  = shift(2) → yesterday's actual
    #   temp_trend_3day = rolling(3).mean().shift(1) → mean of today, yesterday, day before
    row["tmax_yesterday"]  = float(fd["tmax_actual"].iloc[-1])    # today
    row["tmax_2days_ago"]  = float(fd["tmax_actual"].iloc[-2])    # yesterday
    row["temp_trend_3day"] = float(fd["tmax_actual"].iloc[-3:].mean())  # 3-day mean

    # ── B) Bias and anomaly features (carried from today's fd row) ────────────
    # These are rolling means of past errors — tomorrow has no actual yet so
    # we carry today's computed values forward unchanged.
    for col in fc:
        if (col.startswith("bias_3d_") or col.startswith("bias_14d_") or
                col == "forecast_anomaly"):
            if col in today.index and pd.notna(today[col]):
                row[col] = today[col]

    # ── C) Derived features recomputed from tomorrow's NWP values ─────────────
    # model_max_spread: max minus min across all model temperature columns
    tm_cols = [f"temperature_2m_{m}_max" for m in MODELS
               if f"temperature_2m_{m}_max" in row.index]
    if len(tm_cols) >= 2:
        vals = [row[c] for c in tm_cols if pd.notna(row.get(c))]
        row["model_max_spread"] = max(vals) - min(vals) if len(vals) >= 2 else 0.0

    # clear_calm_index and effective_solar use tomorrow's cloud/wind/solar/humidity
    cloud    = row.get("cloud_cover_forecast", 50)
    wind     = row.get("wind_forecast", 5)
    solar    = row.get("solar_ghi_forecast", 0)
    humidity = row.get("humidity_forecast", 70)
    row["clear_calm_index"] = (100 - cloud) / (wind + 1)
    row["effective_solar"]  = solar * (1 - humidity / 200)

    # Calendar features for tomorrow's date
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

def NWPforcast(target_date, city_name):
    script_location = Path(__file__).resolve().parent
    BASE_DIR = script_location / "Data" / target_date
    model_runs_path= BASE_DIR / f"model_runs_Report.csv"
    model_runs_data = pd.read_csv(model_runs_path)
    model_runs_data_filtered = model_runs_data[model_runs_data["city"] == city_name]
    model_runs_data_filtered["mean_max"] = (model_runs_data_filtered["ecmwf_ifs025"] + model_runs_data_filtered["ecmwf_ifs"] + model_runs_data_filtered["gem_seamless"] + model_runs_data_filtered["gfs_seamless"] + model_runs_data_filtered["icon_seamless"] + model_runs_data_filtered["ukmo_seamless"])/6
    print(model_runs_data_filtered)
    # print(ensemble_data_filtered)
    print(f"\n  ──────────────────────────────────────────────\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    station:             str,
    city:                str,
    timezone:            str,
    data_folder:         str   = "mos_data",
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

    Buckets and market prices are loaded automatically from:
        <data_folder>/data/<tomorrow_local_date>/market-<date>.csv
    No need to pass them manually.
    """

    try: pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(f"Unknown timezone '{timezone}'")

    import sys
    log_dir = Path(data_folder) / "logs" / city
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts   = datetime.datetime.now().strftime("%Y-%m-%d_TIme_%H-%M_V3")
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

    # -----------------------------------------------------------------------
    # [5d] CORRECTOR — load persisted real errors OR seed from walk-forward
    #
    # The corrector must track errors from the ACTUAL issued forecasts, not
    # from throwaway mini-models. Each run saves its state; the next run loads
    # it so the corrector accumulates genuine day-by-day operational errors.
    #
    # First run ever (no saved state): fall back to walk-forward seeding so
    # the corrector has something reasonable to start with.
    # -----------------------------------------------------------------------
    print(f"\n[5d] CORRECTOR")
    corrector = load_corrector(city, data_folder,
                               decay=corrector_decay,
                               jump_scale=corrector_jump_scale)
    if corrector is None:
        print(f"  No saved state found — seeding from walk-forward "
              f"(last {corrector_seed_days} days OOS)")
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

    # -----------------------------------------------------------------------
    # Step 6 & 7 – load market data, forecast, bet recommendations
    # The local tomorrow date drives:
    #   a) the section header  b) the market CSV path
    # We use local time (not UTC) because the pipeline skips the last partial
    # METAR day and some cities are still on "today" in UTC at run time.
    # -----------------------------------------------------------------------
    # Derive target date from the last paired training row, not the system clock.
    # fd["date"] is already in local time (daily_metar/daily_model convert it).
    target_date = get_target_date(fd)

    print("\n[MARKET] Loading buckets and prices from market CSV ...")
    buckets, market_prices = load_market_data(city, timezone, target_date, data_folder)

    # Fall back to simulation only when the market file is genuinely absent
    if buckets is None:
        buckets = sorted(set(range(18, 40)))   # broad default — never used in production
        raw = {b: np.random.uniform(0.05, 0.35) for b in buckets}
        tot = sum(raw.values())
        market_prices = {b: round(v / tot, 3) for b, v in raw.items()}
        print("  (Using simulated buckets and prices — no market file found)")

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

    print(f"\n[7] FORECAST FROM NWP {target_date}")
    NWPforcast(target_date, city)

    print(f"\n[8] BET RECOMMENDATION  ({target_date})")
    # bucket_probs works in Celsius — buckets are already in C (load_market_data
    # converts F->C for American cities so everything here is always Celsius)
    probs = bucket_probs(pred["final_forecast"], pred["error_std"], buckets)

    print(f"\n  {'Bucket':<6} {'ML%':>7}  {'Market%':>8}  {'Edge':>7}  Action")
    print("  " + "-" * 46)
    for b in buckets:
        mp   = probs.get(b, 0)
        mkt  = market_prices.get(b, 0)
        e    = mp - mkt
        flag = "  BET" if abs(e) >= 0.05 else ""
        print(f"  {b}C     {mp:>6.1%}   {mkt:>6.1%}   {e:>+6.1%}{flag}")

    bets = bet_recs(probs, market_prices)
    if bets:
        print("\n  Recommended bets:")
        for bt in bets:
            print(f"    {bt['action']} {bt['bucket']}C  "
                  f"edge={bt['edge']}  kelly={bt['kelly']}  {bt['confidence']}")
    else:
        print("\n  No bets above edge threshold.")

    # Save corrector state so the next run has a starting point.
    # IMPORTANT: this saves the pre-forecast state — the corrector does NOT yet
    # know today's actual tmax. After the real tmax is observed, you MUST call:
    #
    #   update_and_save_corrector(
    #       result["corrector"],
    #       forecast_date = target_date,
    #       ml_prediction = result["forecast"]["ml_forecast"],
    #       actual_tmax   = <observed tmax>,
    #       city          = city,
    #       data_folder   = data_folder,
    #   )
    #
    # Without this call the corrector never accumulates real operational errors
    # and will keep re-seeding from mini-models on every run.
    # save_corrector(corrector, city, data_folder)
    print(f"  [CORRECTOR] Reminder: call update_and_save_corrector() once "
          f"today's actual tmax for {target_date} is observed.")

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
            "forecast": pred, "paired_df": pd_, "featured_df": fd, "bets": bets,
            "target_date": target_date}


# =============================================================================
# ENTRY POINTS
# =============================================================================

if __name__ == "__main__":

    all_city_names = [c["name"] for c in cities]

    city_name = sys.argv[1]
    while True:
        # city_name = input("Enter city name: ").strip().lower()
        selected_city = next((c for c in cities if c["name"] == city_name), None)
        if city_name in all_city_names:
            print(f"{city_name} found!")
            break
        else:
            print("City not found. Please try again.")
            city_name = input("Enter city name: ").strip().lower()

    print("These below are the results of V3")
    result = run_pipeline(
        station             = selected_city["station"],
        city                = selected_city["name"],
        timezone            = selected_city["timezone"],
        data_folder         = "mos_data",
        initial_train_days  = 1400,
        run_walk_forward    = False,   # set True for full diagnostic (slow)
        corrector_seed_days = 30,      # increase to 30 for more stable seeding
    )

    print("THESE BELOW ARE THE RESULTS OF V4")

    mos_pipeline_v4.result = mos_pipeline_v4.run_pipeline(
        station             = selected_city["station"],
        city                = selected_city["name"],
        timezone            = selected_city["timezone"],
        data_folder         = "mos_data",
        initial_train_days  = 1400,
        run_walk_forward    = False,   # set True for full diagnostic (slow)
        corrector_seed_days = 30,      # increase to 30 for more stable seeding
    )
