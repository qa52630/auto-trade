"""Compare three training strategies side-by-side on identical walk-forward splits.

  global6    — one model trained on the 6 high-liquidity stocks (baseline)
  per_group  — one model per 族群 (5 models), predicts only its own group's stocks
  per_stock  — one model per stock with ≥ MIN_TRAIN_DAYS data

For each strategy: produces artifacts/preds_<name>.csv, then runs the realistic
NTD backtest (SL=2.5%, TP=3%, fee=0.2355%, 1-lot, 600k/day budget, -10k breaker)
and prints a comparison table.

Usage:
  python3 train_strategies.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import train as T

ROOT = Path(__file__).parent
STOCKS_JSON = ROOT / "stocks.json"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Walk-forward params
TRAIN_DAYS = 30
TEST_DAYS = 5
STEP_DAYS = 5
HOLD_BARS = 6
THRESHOLD = 0.005
MIN_TRAIN_DAYS = 30          # per-stock: need at least this many distinct days
PROBA_SHORT = 0.35

# Realistic backtest params
DAILY_BUDGET = 600_000
DAILY_LOSS_CAP = 10_000
LOT = 1000
SL = 0.025
TP = 0.030
COST = 0.002355
EOD = (13, 25)
MAX_INTRADAY_PCT = 0.07

# Baselines
LIQUID6 = {"6770", "2337", "1802", "2408", "3481", "1815"}


# ---------- shared pipeline ---------- ----------------------------------------

def _build_dataset() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Returns the post-pipeline feature/label dataframe (full 27-stock universe)
    plus a {group_name: [symbols]} dict."""
    stocks = json.loads(STOCKS_JSON.read_text())
    groups = stocks["groups"]
    all_syms = sorted({s for g in groups.values() for s in g})

    df1 = T.load_all()
    df5 = T.resample_5m(df1)
    feats = T.build_features(df5)
    feats = T.add_cross_sectional(feats, STOCKS_JSON)
    feats = feats[feats["symbol"].isin(all_syms)]
    feats = T.label_next(feats, threshold=THRESHOLD, hold_bars=HOLD_BARS)
    feats = feats.dropna(subset=T.FEATURE_COLS + ["fwd_ret"])
    exit_high = feats.groupby("symbol")["high"].shift(-HOLD_BARS)
    exit_low = feats.groupby("symbol")["low"].shift(-HOLD_BARS)
    exit_vol = feats.groupby("symbol")["volume"].shift(-HOLD_BARS)
    feats = feats[
        (feats["volume"] >= 1000) & (exit_vol >= 1000)
        & (feats["high"] != feats["low"]) & (exit_high != exit_low)
        & (feats["fwd_ret"].abs() <= 0.05)
    ]
    feats = feats[feats["label"] != 0].copy()
    feats["label_bin"] = (feats["label"] == 1).astype(int)
    return feats, groups


def _fold_dates(all_dates: list[str]):
    return T.walk_forward_folds(all_dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray | None:
    if len(train_df) < 50 or len(test_df) < 5:
        return None
    if train_df["label_bin"].nunique() < 2:
        return None
    m = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    m.fit(train_df[T.FEATURE_COLS].values, train_df["label_bin"].values)
    return m.predict_proba(test_df[T.FEATURE_COLS].values)[:, 1]


# ---------- strategies --------------------------------------------------------

def strat_global(feats: pd.DataFrame, symbols: set[str], dates: list[str]) -> pd.DataFrame:
    """One model, trained on `symbols` only, predicts only `symbols`."""
    f = feats[feats["symbol"].isin(symbols)]
    out = []
    for i, (tr_d, te_d) in enumerate(_fold_dates(dates), 1):
        tr = f[f["date"].isin(tr_d)]
        te = f[f["date"].isin(te_d)]
        proba = _fit_predict(tr, te)
        if proba is None:
            continue
        sub = te[["symbol", "datetime", "date", "close", "fwd_ret", "label_bin"]].copy()
        sub["proba"] = proba
        sub["position"] = np.where(proba <= PROBA_SHORT, -1, 0)
        sub["fold"] = i
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def strat_per_group(feats: pd.DataFrame, groups: dict[str, list[str]], dates: list[str]) -> pd.DataFrame:
    """One model per group, predicts only its group's stocks."""
    out = []
    folds = list(enumerate(_fold_dates(dates), 1))
    for gname, members in groups.items():
        gset = set(members)
        f = feats[feats["symbol"].isin(gset)]
        if f.empty:
            continue
        for i, (tr_d, te_d) in folds:
            tr = f[f["date"].isin(tr_d)]
            te = f[f["date"].isin(te_d)]
            proba = _fit_predict(tr, te)
            if proba is None:
                continue
            sub = te[["symbol", "datetime", "date", "close", "fwd_ret", "label_bin"]].copy()
            sub["proba"] = proba
            sub["position"] = np.where(proba <= PROBA_SHORT, -1, 0)
            sub["fold"] = i
            sub["group"] = gname
            out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def strat_per_stock(feats: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """One model per stock; skip stocks lacking enough training-day data."""
    out = []
    folds = list(enumerate(_fold_dates(dates), 1))
    for sym, f in feats.groupby("symbol"):
        for i, (tr_d, te_d) in folds:
            tr = f[f["date"].isin(tr_d)]
            te = f[f["date"].isin(te_d)]
            if tr["date"].nunique() < MIN_TRAIN_DAYS:
                continue
            proba = _fit_predict(tr, te)
            if proba is None:
                continue
            sub = te[["symbol", "datetime", "date", "close", "fwd_ret", "label_bin"]].copy()
            sub["proba"] = proba
            sub["position"] = np.where(proba <= PROBA_SHORT, -1, 0)
            sub["fold"] = i
            out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------- realistic backtest ------------------------------------------------

_M1_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}


def _load_1m(day: str, sym: str) -> pd.DataFrame | None:
    k = (day, sym)
    if k in _M1_CACHE:
        return _M1_CACHE[k]
    base = ROOT / "data" / day[:7]
    for prefix in ("masterlink", "taishin"):
        p = base / f"{prefix}_{day}_{sym}_1m.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "datetime" in df.columns:
            df["dt"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        else:
            df["dt"] = pd.to_datetime(day + " " + df["time"])
        cols = ["dt", "open", "high", "low", "close"]
        if "ref_price" in df.columns:
            cols.append("ref_price")
        _M1_CACHE[k] = df[cols].sort_values("dt").reset_index(drop=True)
        return _M1_CACHE[k]
    _M1_CACHE[k] = None
    return None


def realistic_pnl(preds: pd.DataFrame, label: str) -> dict:
    sigs = preds[preds["position"] != 0].copy()
    if sigs.empty:
        return {"label": label, "fills": 0, "win_rate": 0, "total_pnl": 0, "mdd": 0}
    sigs["signal_dt"] = pd.to_datetime(sigs["datetime"]).dt.tz_localize(None)
    sigs = sigs.sort_values("signal_dt").reset_index(drop=True)

    budget = defaultdict(float)
    daily_pnl = defaultdict(float)
    breaker = set()
    rows = []
    skip = defaultdict(int)

    for _, s in sigs.iterrows():
        sym = str(s["symbol"])
        d = s["signal_dt"].strftime("%Y-%m-%d")
        if d in breaker:
            skip["breaker"] += 1; continue
        m1 = _load_1m(d, sym)
        if m1 is None:
            skip["nodata"] += 1; continue
        # 7% filter
        if "ref_price" in m1.columns:
            rp = m1["ref_price"].dropna()
            rp = rp[rp > 0]
            if len(rp):
                pct = (float(s["close"]) - float(rp.iloc[0])) / float(rp.iloc[0])
                if pct >= MAX_INTRADAY_PCT:
                    skip["pct"] += 1; continue
        after = m1[m1["dt"] >= s["signal_dt"] + pd.Timedelta(minutes=5)]
        if after.empty:
            skip["nobar"] += 1; continue
        ep = float(after.iloc[0]["open"])
        if ep <= 0:
            skip["badpx"] += 1; continue
        notional = ep * LOT
        if budget[d] + notional > DAILY_BUDGET:
            skip["budget"] += 1; continue
        sl_px = ep * (1 + SL); tp_px = ep * (1 - TP); exit_p = None; reason = None
        for _, b in after.iloc[1:].iterrows():
            if (b["dt"].hour, b["dt"].minute) >= EOD:
                exit_p, reason = float(b["open"]), "eod"; break
            if b["high"] >= sl_px:
                exit_p, reason = sl_px, "sl"; break
            if b["low"] <= tp_px:
                exit_p, reason = tp_px, "tp"; break
        if exit_p is None:
            exit_p, reason = float(after.iloc[-1]["close"]), "last"
        gross = (ep - exit_p) / ep
        net = gross - COST
        pnl = notional * net
        budget[d] += notional
        daily_pnl[d] += pnl
        if daily_pnl[d] <= -DAILY_LOSS_CAP:
            breaker.add(d)
        rows.append({"date": d, "symbol": sym, "fold": int(s["fold"]),
                     "pnl": pnl, "reason": reason})

    ex = pd.DataFrame(rows)
    if ex.empty:
        return {"label": label, "fills": 0, "win_rate": 0, "total_pnl": 0, "mdd": 0}
    ex["win"] = ex["pnl"] > 0
    cum = ex["pnl"].cumsum()
    peak = cum.cummax()
    w = ex.loc[ex["win"], "pnl"]; l = ex.loc[~ex["win"], "pnl"]
    return {
        "label": label,
        "signals": int(len(sigs)),
        "fills": int(len(ex)),
        "skipped_budget": skip["budget"],
        "win_rate": round(float(ex["win"].mean()) * 100, 1),
        "total_pnl": int(round(ex["pnl"].sum())),
        "avg_pnl": int(round(ex["pnl"].mean())),
        "pf": round(abs(w.sum() / l.sum()), 2) if len(l) and l.sum() != 0 else 0,
        "mdd": int(round((cum - peak).min())),
        "fold_pnls": {int(k): int(round(v)) for k, v in ex.groupby("fold")["pnl"].sum().items()},
        "n_symbols": int(ex["symbol"].nunique()),
        "by_symbol_pnl": ex.groupby("symbol")["pnl"].sum().round(0).astype(int).to_dict(),
    }


def main() -> int:
    print("[1/3] building dataset (full 27-stock universe)...", file=sys.stderr)
    feats, groups = _build_dataset()
    all_dates = sorted(feats["date"].unique())
    print(f"      rows={len(feats):,}  symbols={feats['symbol'].nunique()}  days={len(all_dates)}",
          file=sys.stderr)

    print("[2/3] training 3 strategies...", file=sys.stderr)
    setups = {
        "global6":   strat_global(feats, LIQUID6, all_dates),
        "per_group": strat_per_group(feats, groups, all_dates),
        "per_stock": strat_per_stock(feats, all_dates),
    }
    for name, df in setups.items():
        p = ARTIFACTS / f"preds_{name}.csv"
        df.to_csv(p, index=False)
        n_sig = int((df["position"] != 0).sum()) if len(df) else 0
        print(f"      {name:<10} rows={len(df):,}  signals={n_sig}  symbols={df['symbol'].nunique() if len(df) else 0}  → {p.name}",
              file=sys.stderr)

    print("[3/3] realistic backtest...", file=sys.stderr)
    summaries = {name: realistic_pnl(df, name) for name, df in setups.items()}

    print("\n" + "=" * 96)
    print(f"{'strategy':<12} {'signals':>8} {'fills':>6} {'symbols':>8} {'win%':>6} "
          f"{'total NTD':>11} {'本金 %':>7} {'avg':>6} {'PF':>5} {'MDD':>9}")
    print("-" * 96)
    for name, s in summaries.items():
        pct = s["total_pnl"] / 300_000 * 100 if s["fills"] else 0
        print(f"{s['label']:<12} {s.get('signals',0):>8} {s['fills']:>6} {s.get('n_symbols',0):>8} "
              f"{s['win_rate']:>6.1f} {s['total_pnl']:>+11,} {pct:>+6.1f}% "
              f"{s.get('avg_pnl',0):>+6,} {s.get('pf',0):>5.2f} {s['mdd']:>+9,}")
    print("=" * 96)

    print("\n=== 各 fold PnL ===")
    print(f"{'strategy':<12} " + "  ".join(f"fold{i}" for i in range(1, 8)))
    for name, s in summaries.items():
        fp = s.get("fold_pnls", {})
        cells = "  ".join(f"{fp.get(i, 0):>+6,}" for i in range(1, 8))
        print(f"{s['label']:<12} {cells}")

    print("\n=== per_stock 各標的 PnL (含正負) ===")
    bs = summaries.get("per_stock", {}).get("by_symbol_pnl", {})
    rows = sorted(bs.items(), key=lambda kv: -kv[1])
    for sym, pnl in rows:
        print(f"  {sym}: {pnl:>+8,}")

    # --- Train + save FINAL models on ALL history (for live use) ---
    print("\n[final] training & saving live-use models on all-time data...", file=sys.stderr)
    recent = feats   # use full history; more data = more stocks pass row threshold
    pg_dir = ARTIFACTS / "models_per_group"; pg_dir.mkdir(exist_ok=True)
    ps_dir = ARTIFACTS / "models_per_stock"; ps_dir.mkdir(exist_ok=True)
    pg_meta = {}
    for gname, members in groups.items():
        sub = recent[recent["symbol"].isin(members)]
        if len(sub) < 50 or sub["label_bin"].nunique() < 2:
            continue
        m = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        )
        m.fit(sub[T.FEATURE_COLS].values, sub["label_bin"].values)
        path = pg_dir / f"{gname}.joblib"
        joblib.dump(m, path)
        pg_meta[gname] = {"members": members, "train_rows": len(sub),
                          "train_days": int(sub["date"].nunique())}
        print(f"      per_group/{gname}: rows={len(sub)}  → {path.name}", file=sys.stderr)
    (pg_dir / "_meta.json").write_text(json.dumps(pg_meta, ensure_ascii=False, indent=2))

    ps_meta = {}
    for sym, sub in recent.groupby("symbol"):
        if sub["date"].nunique() < MIN_TRAIN_DAYS:
            continue
        if len(sub) < 50 or sub["label_bin"].nunique() < 2:
            continue
        m = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        )
        m.fit(sub[T.FEATURE_COLS].values, sub["label_bin"].values)
        path = ps_dir / f"{sym}.joblib"
        joblib.dump(m, path)
        ps_meta[sym] = {"train_rows": len(sub), "train_days": int(sub["date"].nunique())}
        print(f"      per_stock/{sym}: rows={len(sub)}  → {path.name}", file=sys.stderr)
    (ps_dir / "_meta.json").write_text(json.dumps(ps_meta, ensure_ascii=False, indent=2))

    print(f"\nartifacts → {ARTIFACTS}/preds_*.csv, models_per_group/, models_per_stock/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
