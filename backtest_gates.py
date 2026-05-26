"""Experiment: add two entry gates to the short-only strategies and compare
against the current baseline on identical walk-forward splits.

Gates (both OFF = current production behaviour):
  A) trend gate     — skip a short if the stock is showing upward momentum:
                      ret_6 >= R  OR  ma_20 <= -M  (price above its 20MA).
  B) group-weak gate — within the same (sector, datetime), among the stocks that
                      would be shorted, keep ONLY the one with the lowest
                      rel_str_6, and require rel_str_6 < 0.

This script NEVER retrains or overwrites the live models. It only reuses the
read-only dataset / fold / backtest helpers from train_strategies.

Usage:
  python3 backtest_gates.py
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
import pandas as pd

import train as T
import train_strategies as TS

# Gate thresholds (single combo here; tweak to sweep)
TREND_RET6 = 0.005    # ret_6 >= this  -> skip short (upward 30-min momentum)
TREND_MA20 = 0.003    # ma_20 <= -this -> skip short (close above 20MA = strong)

GATE_COLS = ["ret_6", "ma_20", "co_gap", "rel_str_6", "sector"]


def _fold_preds(strategy: str, feats: pd.DataFrame, groups, dates) -> pd.DataFrame:
    """Run walk-forward for one strategy, returning per-bar predictions WITH the
    gate columns attached (so we can apply gates afterwards)."""
    keep = ["symbol", "datetime", "date", "close", "fwd_ret", "label_bin"] + GATE_COLS
    out = []
    folds = list(enumerate(TS._fold_dates(dates), 1))

    def _run(f, extra_group=None):
        for i, (tr_d, te_d) in folds:
            tr = f[f["date"].isin(tr_d)]
            te = f[f["date"].isin(te_d)]
            proba = TS._fit_predict(tr, te)
            if proba is None:
                continue
            sub = te[keep].copy()
            sub["proba"] = proba
            sub["fold"] = i
            if extra_group is not None:
                sub["group"] = extra_group
            out.append(sub)

    if strategy == "global6":
        _run(feats[feats["symbol"].isin(TS.LIQUID6)])
    elif strategy == "per_group":
        for gname, members in groups.items():
            f = feats[feats["symbol"].isin(set(members))]
            if not f.empty:
                _run(f, extra_group=gname)
    elif strategy == "per_stock":
        for sym, f in feats.groupby("symbol"):
            for i, (tr_d, te_d) in folds:
                tr = f[f["date"].isin(tr_d)]
                te = f[f["date"].isin(te_d)]
                if tr["date"].nunique() < TS.MIN_TRAIN_DAYS:
                    continue
                proba = TS._fit_predict(tr, te)
                if proba is None:
                    continue
                sub = te[keep].copy()
                sub["proba"] = proba
                sub["fold"] = i
                out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _apply_gates(preds: pd.DataFrame, trend: bool, group_weak: bool) -> pd.DataFrame:
    """Return a copy of preds with a `position` column reflecting the gates."""
    p = preds.copy()
    short = p["proba"] <= TS.PROBA_SHORT          # baseline short candidates

    if trend:
        strong = (p["ret_6"] >= TREND_RET6) | (p["ma_20"] <= -TREND_MA20)
        short = short & ~strong

    if group_weak:
        # among current short candidates, within (sector, datetime) keep only the
        # weakest (lowest rel_str_6) and require rel_str_6 < 0
        cand = p[short & (p["rel_str_6"] < 0)].copy()
        if cand.empty:
            keep_idx = set()
        else:
            idx = cand.groupby(["sector", "datetime"])["rel_str_6"].idxmin()
            keep_idx = set(idx.values)
        short = p.index.isin(keep_idx)
        short = pd.Series(short, index=p.index)

    p["position"] = np.where(short, -1, 0)
    return p


def main() -> int:
    print("[1/3] building dataset (read-only, no model overwrite)...", file=sys.stderr)
    feats, groups = TS._build_dataset()
    dates = sorted(feats["date"].unique())
    print(f"      rows={len(feats):,}  symbols={feats['symbol'].nunique()}  days={len(dates)}",
          file=sys.stderr)

    variants = {
        "baseline":   (False, False),
        "+trend":     (True, False),
        "+groupweak": (False, True),
        "+both":      (True, True),
    }

    results = {}
    for strat in ("global6", "per_group", "per_stock"):
        print(f"[2/3] walk-forward: {strat} ...", file=sys.stderr)
        preds = _fold_preds(strat, feats, groups, dates)
        if preds.empty:
            print(f"      {strat}: no preds", file=sys.stderr)
            continue
        for vname, (tr, gw) in variants.items():
            pv = _apply_gates(preds, tr, gw)
            summ = TS.realistic_pnl(pv, f"{strat}/{vname}")
            results[(strat, vname)] = summ

    print("\n" + "=" * 104)
    print(f"trend_ret6={TREND_RET6}  trend_ma20={TREND_MA20}")
    print("=" * 104)
    hdr = (f"{'strategy/variant':<22} {'signals':>8} {'fills':>6} {'syms':>5} "
           f"{'win%':>6} {'total NTD':>11} {'avg':>6} {'PF':>5} {'MDD':>9}")
    for strat in ("global6", "per_group", "per_stock"):
        print("-" * 104)
        print(hdr)
        for vname in variants:
            s = results.get((strat, vname))
            if not s:
                continue
            print(f"{s['label']:<22} {s.get('signals',0):>8} {s['fills']:>6} "
                  f"{s.get('n_symbols',0):>5} {s['win_rate']:>6.1f} "
                  f"{s['total_pnl']:>+11,} {s.get('avg_pnl',0):>+6,} "
                  f"{s.get('pf',0):>5.2f} {s['mdd']:>+9,}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
