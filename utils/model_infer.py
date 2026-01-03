import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from tensorflow.keras.models import load_model
import joblib

_MODEL = None
_SCALER = None
_FEATS = None
_NSTEPS = None

def _smart_load_scaler(path: str):
    obj = joblib.load(path)
    return obj

def _pick_feature_scaler(obj):
    if hasattr(obj, "transform"):
        return obj, None, 6
    if isinstance(obj, dict):
        sc = obj.get("x_scaler") or obj.get("scaler") or obj.get("feature_scaler")
        feats = obj.get("feature_cols")
        nsteps = int(obj.get("n_steps", 6))
        if sc is None:
            raise ValueError("Scaler tidak ditemukan di dict.")
        return sc, feats, nsteps
    raise ValueError("Format scaler tidak dikenali.")

def _load_artifacts(model_path: str = "models/best_model_fixed.h5",
                    scaler_path: str = "models/scaler_bundle_LOG.pkl"):
    global _MODEL, _SCALER, _FEATS, _NSTEPS, _Y_LOG, _Y_MU, _Y_SD
    if _MODEL is None:
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
        _MODEL = load_model(str(mp), compile=False)
    if _SCALER is None:
        raw = _smart_load_scaler(scaler_path)
        sc, feats, nsteps = _pick_feature_scaler(raw)
        _SCALER, _FEATS, _NSTEPS = sc, feats, nsteps

        try:
            from collections.abc import Mapping
            if isinstance(raw, Mapping):
                _Y_LOG = bool(raw.get("y_log", False))
                _Y_MU  = float(raw.get("y_mu", 0.0))
                _Y_SD  = float(raw.get("y_sd", 1.0))
        except Exception:
            _Y_LOG, _Y_MU, _Y_SD = False, 0.0, 1.0

def _month_sin_cos(idx: pd.DatetimeIndex) -> pd.DataFrame:
    m = idx.month.values
    return pd.DataFrame({
        "month_sin": np.sin(2*np.pi*m/12),
        "month_cos": np.cos(2*np.pi*m/12),
    }, index=idx)

def _mode_safe(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return None
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else None

def _to_monthly(df_prod: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    d = df_prod.copy()
    d["Tanggal"] = pd.to_datetime(d["Tanggal"], errors="coerce")
    d = d.dropna(subset=["Tanggal"]).sort_values("Tanggal")
    s_daily = d.groupby("Tanggal")["Jumlah Terjual"].sum()
    y_m = s_daily.resample("MS").sum()
    promo_m = d.groupby(pd.Grouper(key="Tanggal", freq="MS"))["Promotion"].agg(_mode_safe) if "Promotion" in d.columns else None
    holi_m = d.groupby(pd.Grouper(key="Tanggal", freq="MS"))["Holiday"].agg(_mode_safe) if "Holiday" in d.columns else None
    return y_m, promo_m, holi_m

def _build_features(y_m: pd.Series,
                    promo_m: Optional[pd.Series],
                    holi_m: Optional[pd.Series],
                    lags: int,
                    ma: int = 3) -> pd.DataFrame:
    g = pd.DataFrame({"y": y_m})
    mc = _month_sin_cos(g.index)
    g = pd.concat([g, mc], axis=1)
    for i in range(1, lags+1):
        g[f"lag{i}"] = g["y"].shift(i)
    g[f"ma{ma}"] = g["y"].rolling(ma).mean()
    for k in ["A","B","C","D"]:
        if promo_m is None:
            g[f"promo{k}"] = 0
        else:
            g[f"promo{k}"] = (promo_m.reindex(g.index).astype(str) == k).astype(int)
    for k in [1,2,3,4]:
        if holi_m is None:
            g[f"holi{k}"] = 0
        else:
            g[f"holi{k}"] = (holi_m.reindex(g.index).astype(str) == str(k)).astype(int)
    g = g.dropna()
    return g

def _align_feature_order(df_feats: pd.DataFrame, feature_cols: Optional[list]) -> pd.DataFrame:
    if feature_cols is None:
        return df_feats
    cols = [c for c in feature_cols if c in df_feats.columns]
    missing = set(feature_cols) - set(cols)
    for m in missing:
        df_feats[m] = 0.0
    return df_feats[feature_cols]

def _make_sequence(X_hist: np.ndarray, n_steps: int) -> np.ndarray:
    x = X_hist[-1:, :]
    return x.reshape(1, 1, x.shape[1])

def predict_with_lstm_for_product(df_all: pd.DataFrame, product_name: str, horizon: int,
                                  promo_code: str | None = None,
                                  holi_code: int | None = None) -> List[int]:
    _load_artifacts()
    sub = df_all[df_all["Nama Produk"] == product_name].copy()
    if sub.empty:
        raise ValueError(f"Tidak ada data untuk produk: {product_name}")
    y_m, promo_m, holi_m = _to_monthly(sub)
    feats = _build_features(y_m, promo_m, holi_m, lags=_NSTEPS, ma=3)
    if feats.empty:
        raise ValueError("Fitur kosong setelah konstruksi. Periksa data produk.")
    X_hist_df = feats.drop(columns=["y"])
    X_hist_df = _align_feature_order(X_hist_df, _FEATS)
    X_hist = X_hist_df.values.astype(float)
    X_scaled = _SCALER.transform(X_hist)

    preds: List[int] = []
    y_hist = feats["y"].astype(float).copy()
    current_month = feats.index.max() + pd.offsets.MonthBegin(1)
    X_seq = _make_sequence(X_scaled, _NSTEPS)

    for _ in range(horizon):
        yhat = float(_MODEL.predict(X_seq, verbose=0).squeeze())
        if '_Y_LOG' in globals() and _Y_LOG:
            yhat = np.expm1(yhat * _Y_SD + _Y_MU)
        yint = int(round(max(0, yhat)))
        preds.append(yint)

        next_idx = current_month
        y_hist.loc[next_idx] = yint

        row = {"y": yint}
        for i in range(1, _NSTEPS+1):
            row[f"lag{i}"] = y_hist.iloc[-i] if len(y_hist) >= i else y_hist.iloc[-1]
        row["ma3"] = pd.Series(y_hist).rolling(3).mean().iloc[-1] if len(y_hist) >= 3 else float(y_hist.iloc[-1])
        row["month_sin"] = np.sin(2*np.pi*next_idx.month/12)
        row["month_cos"] = np.cos(2*np.pi*next_idx.month/12)

        for k in ["A","B","C","D"]:
            row[f"promo{k}"] = 1 if (promo_code == k) else 0
        for k in [1,2,3,4]:
            row[f"holi{k}"] = 1 if (holi_code == k) else 0

        x_next_df = pd.DataFrame([row]).drop(columns=["y"])
        x_next_df = _align_feature_order(x_next_df, _FEATS)
        x_next = x_next_df.values.astype(float)
        x_next_scaled = _SCALER.transform(x_next)
        X_seq = x_next_scaled.reshape(1, 1, x_next_scaled.shape[1])

        current_month = current_month + pd.offsets.MonthBegin(1)

    return preds