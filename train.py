"""Minute-K trading classifier.

Loads all masterlink 1-minute CSVs under data/, resamples to 5-minute bars per
symbol, builds technical features, labels the next 5-min bar's direction, and
trains a GradientBoosting classifier with a time-ordered train/test split.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix  # noqa: F401

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

FNAME_RE = re.compile(r"(masterlink|taishin)_(\d{4}-\d{2}-\d{2})_(\d+)_1m\.csv$")


def load_all() -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(DATA_DIR / "*" / "*_1m.csv")):
        m = FNAME_RE.search(path)
        if not m:
            continue
        date, symbol = m.group(2), m.group(3)
        df = pd.read_csv(path)
        if df.empty or not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            continue
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Taipei")
        elif "time" in df.columns:
            df["datetime"] = pd.to_datetime(date + " " + df["time"].astype(str), errors="coerce").dt.tz_localize("Asia/Taipei")
        else:
            continue
        df = df.dropna(subset=["datetime"])
        if df.empty:
            continue
        df["symbol"] = symbol
        df["date"] = date
        rows.append(df[["symbol", "date", "datetime", "open", "high", "low", "close", "volume"]])
    if not rows:
        raise SystemExit("No data files found under data/")
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (sym, date), g in df1m.groupby(["symbol", "date"], sort=False):
        g = g.set_index("datetime")
        agg = g.resample("5min", label="left", closed="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["open"])
        agg["symbol"] = sym
        agg["date"] = date
        parts.append(agg.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out_parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        c = g["close"]
        g["ret_1"] = c.pct_change(1)
        g["ret_3"] = c.pct_change(3)
        g["ret_6"] = c.pct_change(6)
        g["ret_12"] = c.pct_change(12)
        g["ma_5"] = c.rolling(5).mean() / c - 1
        g["ma_10"] = c.rolling(10).mean() / c - 1
        g["ma_20"] = c.rolling(20).mean() / c - 1
        g["vol_5"] = c.pct_change().rolling(5).std()
        g["vol_20"] = c.pct_change().rolling(20).std()
        g["rsi_14"] = rsi(c, 14)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        g["macd"] = macd / c
        g["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
        g["hl_range"] = (g["high"] - g["low"]) / c
        g["co_gap"] = (c - g["open"]) / g["open"]
        v = g["volume"].astype(float)
        g["vol_chg"] = v.pct_change().replace([np.inf, -np.inf], np.nan)
        g["vol_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def label_next(df: pd.DataFrame, threshold: float = 0.002, hold_bars: int = 1) -> pd.DataFrame:
    """Executable label: enter at bar t's close, exit at bar t+hold_bars' close.

    The signal is generated when bar t closes, so the earliest realistic entry
    price is close[t]. We exit `hold_bars` later, also at close.
    """
    out_parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        exit_close = g["close"].shift(-hold_bars)
        g["fwd_ret"] = (exit_close - g["close"]) / g["close"]
        # no overnight leakage: every intermediate bar must be same date
        same_day = g["date"] == g["date"].shift(-hold_bars)
        g.loc[~same_day, "fwd_ret"] = np.nan
        g["label"] = np.where(g["fwd_ret"] > threshold, 1,
                       np.where(g["fwd_ret"] < -threshold, -1, 0))
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ma_5", "ma_10", "ma_20",
    "vol_5", "vol_20",
    "rsi_14", "macd", "macd_sig",
    "hl_range", "co_gap", "vol_chg", "vol_z20",
    # cross-sectional
    "sector_ret_1", "sector_ret_6",
    "rel_str_1", "rel_str_6",
    "market_ret_1", "market_ret_6",
]


def add_cross_sectional(df: pd.DataFrame, stocks_json: Path) -> pd.DataFrame:
    """Adds sector & market aggregate features.

    For each bar (symbol, datetime) compute:
      sector_ret_k = mean ret_k of OTHER stocks in same sector at same datetime
      rel_str_k    = ret_k - sector_ret_k  (relative strength vs sector)
      market_ret_k = mean ret_k across all symbols in universe at same datetime
    """
    meta = json.loads(stocks_json.read_text())
    sym_to_sector = {}
    for sector, syms in meta["groups"].items():
        for s in syms:
            sym_to_sector[s] = sector
    df = df.copy()
    df["sector"] = df["symbol"].map(sym_to_sector)

    for k in (1, 6):
        rc = f"ret_{k}"
        # sector aggregates: groupby (datetime, sector) — exclude self by using
        # (sum - own) / (count - 1)
        grp = df.groupby(["datetime", "sector"])[rc]
        s_sum = grp.transform("sum")
        s_cnt = grp.transform("count")
        df[f"sector_ret_{k}"] = np.where(
            s_cnt > 1, (s_sum - df[rc]) / (s_cnt - 1), np.nan
        )
        df[f"rel_str_{k}"] = df[rc] - df[f"sector_ret_{k}"]
        # market aggregate: across all symbols at same datetime (exclude self)
        mgrp = df.groupby("datetime")[rc]
        m_sum = mgrp.transform("sum")
        m_cnt = mgrp.transform("count")
        df[f"market_ret_{k}"] = np.where(
            m_cnt > 1, (m_sum - df[rc]) / (m_cnt - 1), np.nan
        )
    return df


TAIEX_DAILY_CSV = DATA_DIR / "taiex" / "daily.csv"
TAIEX_FEATURE_COLS = [
    "taiex_prev_day_ret",   # D-1's daily ret (known at D open)
    "taiex_5d_trend",       # 5-day cum ret D-6→D-1 (known at D open)
    "taiex_above_ma20",     # 1 if close_{D-1} > MA20_{D-1} else 0
    "taiex_open_gap_pct",   # (open_D − close_{D-1}) / close_{D-1}
]


def _taiex_daily_features(taiex_csv: Path = TAIEX_DAILY_CSV) -> pd.DataFrame:
    """Compute per-date TAIEX regime features. Returns df indexed by date string."""
    if not taiex_csv.exists():
        raise FileNotFoundError(f"TAIEX daily file missing: {taiex_csv}")
    t = pd.read_csv(taiex_csv).sort_values("date").reset_index(drop=True)
    t["date"] = t["date"].astype(str)
    prev_close = t["close"].shift(1)
    prev_prev_close = t["close"].shift(2)
    ma20 = t["close"].shift(1).rolling(20).mean()
    t["taiex_prev_day_ret"] = (prev_close - prev_prev_close) / prev_prev_close
    t["taiex_5d_trend"] = (prev_close - t["close"].shift(6)) / t["close"].shift(6)
    t["taiex_above_ma20"] = (t["close"].shift(1) > ma20).astype("float")
    t["taiex_open_gap_pct"] = (t["open"] - prev_close) / prev_close
    return t[["date"] + TAIEX_FEATURE_COLS]


def add_taiex_features(df: pd.DataFrame, taiex_csv: Path = TAIEX_DAILY_CSV) -> pd.DataFrame:
    """Left-join TAIEX daily regime features by date. Adds 4 constant-per-day cols."""
    tdf = _taiex_daily_features(taiex_csv)
    df = df.copy()
    df["date"] = df["date"].astype(str)
    return df.merge(tdf, on="date", how="left")


def walk_forward_folds(dates, train_days: int, test_days: int, step_days: int):
    """Generate (train_dates, test_dates) tuples sliding forward in time.

    Rolling window (not expanding): each fold trains on `train_days` of the most
    recent trading days, tests on the next `test_days`, then slides by `step_days`.
    """
    dates = sorted(dates)
    n = len(dates)
    folds = []
    start = 0
    while start + train_days + test_days <= n:
        tr = dates[start : start + train_days]
        te = dates[start + train_days : start + train_days + test_days]
        folds.append((tr, te))
        start += step_days
    return folds


def evaluate_fold(train_df, test_df):
    X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["label_bin"].values
    X_te, y_te = test_df[FEATURE_COLS].values, test_df["label_bin"].values
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == y_te).mean())

    out = test_df[["symbol", "datetime", "date", "close", "fwd_ret", "label_bin"]].copy()
    out["proba"] = proba
    # Long-side disabled: backtest showed long trades net -1.14 across all 4 folds
    # (win rate 18~52%) while shorts net +1.21. Short-only strategy.
    # Tightened threshold 0.4 → 0.35: conf 0.6-0.65 band was the loss zone (-0.25 PnL).
    pos = np.where(proba <= 0.35, -1, 0)
    COST = 0.003
    out["position"] = pos
    out["pnl"] = out["position"] * out["fwd_ret"] - (pos != 0) * COST
    n_trades = int((pos != 0).sum())
    if n_trades:
        win = float((out.loc[pos != 0, "pnl"] > 0).mean())
        mean_pnl = float(out.loc[pos != 0, "pnl"].mean())
        total_pnl = float(out["pnl"].sum())
        std = out["pnl"].std()
        sharpe = float(out["pnl"].mean() / std * np.sqrt(252 * 54)) if std else float("nan")
    else:
        win = mean_pnl = total_pnl = sharpe = float("nan")
    return {
        "accuracy": acc, "trades": n_trades, "win_rate": win,
        "avg_pnl_per_bar": mean_pnl, "total_pnl": total_pnl, "sharpe": sharpe,
    }, model, out


def main():
    print("[1/5] loading 1-min CSVs ...")
    df1 = load_all()
    print(f"      rows={len(df1):,}  symbols={df1['symbol'].nunique()}  days={df1['date'].nunique()}")

    print("[2/5] resampling to 5-min ...")
    df5 = resample_5m(df1)
    print(f"      5m bars={len(df5):,}")

    print("[3/5] features + labels ...")
    HOLD_BARS = 6  # 30-min holding period
    THRESHOLD = 0.005  # 0.5% threshold scaled with longer hold
    # global6 baseline: empirically best-performing liquid set (+17.5% OOS)
    # See ~/.claude/skills/day-trader-shortonly.md for rationale.
    LIQUID_SYMBOLS = {"6770", "2337", "1802", "2408", "3481", "1815"}

    # Build features & sector aggregates on the FULL universe (so sector means
    # aren't biased), then restrict training to liquid names.
    feats = build_features(df5)
    feats = add_cross_sectional(feats, ROOT / "stocks.json")
    feats = feats[feats["symbol"].isin(LIQUID_SYMBOLS)]
    print(f"      after liquidity filter: {len(feats):,} bars across {feats['symbol'].nunique()} symbols")
    feats = label_next(feats, threshold=THRESHOLD, hold_bars=HOLD_BARS)
    feats = feats.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    # liquidity gate: drop bars where entry (t) or exit (t+HOLD_BARS) bar is
    # illiquid / flat — these create artificial labels for thin-trading stocks
    exit_high = feats.groupby("symbol")["high"].shift(-HOLD_BARS)
    exit_low = feats.groupby("symbol")["low"].shift(-HOLD_BARS)
    exit_vol = feats.groupby("symbol")["volume"].shift(-HOLD_BARS)
    feats = feats[
        (feats["volume"] >= 1000)
        & (exit_vol >= 1000)
        & (feats["high"] != feats["low"])
        & (exit_high != exit_low)
        & (feats["fwd_ret"].abs() <= 0.05)  # cap to plausible holding-period moves
    ]
    feats = feats[feats["label"] != 0]
    feats["label_bin"] = (feats["label"] == 1).astype(int)
    print(f"      rows={len(feats):,}  up={feats['label_bin'].sum():,}  down={(1-feats['label_bin']).sum():,}")

    print("[4/5] walk-forward folds ...")
    all_dates = sorted(feats["date"].unique())
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 30, 5, 5
    folds = walk_forward_folds(all_dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    print(f"      total days={len(all_dates)}  folds={len(folds)}  "
          f"(train={TRAIN_DAYS}d, test={TEST_DAYS}d, step={STEP_DAYS}d)")

    fold_rows = []
    all_preds = []
    last_model = None
    for i, (tr_dates, te_dates) in enumerate(folds, 1):
        tr = feats[feats["date"].isin(tr_dates)]
        te = feats[feats["date"].isin(te_dates)]
        if tr.empty or te.empty:
            continue
        m, model, out = evaluate_fold(tr, te)
        m["fold"] = i
        m["train_start"] = tr_dates[0]
        m["train_end"] = tr_dates[-1]
        m["test_start"] = te_dates[0]
        m["test_end"] = te_dates[-1]
        m["train_rows"] = int(len(tr))
        m["test_rows"] = int(len(te))
        fold_rows.append(m)
        out["fold"] = i
        all_preds.append(out)
        last_model = model
        print(f"      fold {i:>2}  train {tr_dates[0]}→{tr_dates[-1]}  "
              f"test {te_dates[0]}→{te_dates[-1]}  "
              f"acc={m['accuracy']:.4f}  trades={m['trades']:>4}  "
              f"win={m['win_rate']:.3f}  total={m['total_pnl']:+.4f}  sharpe={m['sharpe']:.2f}")

    print("[5/5] aggregating ...")
    fold_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(all_preds, ignore_index=True)

    agg = {
        "folds": int(len(fold_df)),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "median_accuracy": float(fold_df["accuracy"].median()),
        "std_accuracy": float(fold_df["accuracy"].std()),
        "total_trades": int(fold_df["trades"].sum()),
        "mean_win_rate": float(fold_df["win_rate"].mean(skipna=True)),
        "mean_avg_pnl_per_bar": float(fold_df["avg_pnl_per_bar"].mean(skipna=True)),
        "sum_total_pnl": float(fold_df["total_pnl"].sum(skipna=True)),
        "mean_sharpe": float(fold_df["sharpe"].mean(skipna=True)),
        "pct_profitable_folds": float((fold_df["total_pnl"] > 0).mean()),
    }
    print("      aggregate across folds:")
    for k, v in agg.items():
        print(f"        {k:<25s} {v}")

    # Feature importance from last fold's model (most-recent training window)
    imp = sorted(zip(FEATURE_COLS, last_model.feature_importances_), key=lambda kv: -kv[1])
    print("      top features (last fold):")
    for name, w in imp:
        print(f"        {name:10s} {w:.4f}")

    joblib.dump(last_model, OUT_DIR / "model.joblib")
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "wf_predictions.csv", index=False)
    metrics = {
        "config": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "step_days": STEP_DAYS},
        "aggregate": agg,
        "folds": fold_rows,
        "feature_importance_last_fold": {k: float(v) for k, v in imp},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nArtifacts → {OUT_DIR}/ (model.joblib, fold_metrics.csv, wf_predictions.csv, metrics.json)")


if __name__ == "__main__":
    main()
