from pathlib import Path
import pandas as pd
import numpy as np

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


def corrector_state_path(city: str, data_folder: str = "Data") -> Path:
    """Path to the JSON file that persists the corrector state between runs."""

    script_dir = Path(__file__).resolve().parent
    data_folder = Path(data_folder)
    full_path = data_folder / "corrector-folder-v4"
    return script_dir / full_path / f"corrector_{city.lower()}.json"


def save_corrector(corrector: RealtimeBiasCorrector,
                   city: str, data_folder: str = "Data") -> None:
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


def load_corrector(city: str, data_folder: str = "Data",
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
    # return corr
    return None


def update_and_save_corrector(corrector: RealtimeBiasCorrector,
                               forecast_date: str,
                               ml_prediction: float,
                               actual_tmax: float,
                               city: str,
                               data_folder: str = "Data") -> None:
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