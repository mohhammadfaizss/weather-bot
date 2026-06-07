import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config import MODELS
from features import safe_fillna, feature_cols
from corrector import RealtimeBiasCorrector


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
