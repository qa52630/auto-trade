"""Threshold sweep for the trend gate, on top of the per-model gate config Ben
approved:

  global6   -> +both   (trend gate + group-weak gate)
  per_group -> +trend
  per_stock -> +trend

The expensive walk-forward predictions are computed ONCE per strategy (gates are
applied post-prediction, so they don't depend on the thresholds). Then every
(ret_6, ma_20) threshold combo just re-applies the gate + re-runs the NTD
backtest. NEVER retrains or overwrites live models.

Usage:
  python3 backtest_sweep.py
"""
from __future__ import annotations

import sys

import backtest_gates as BG
import train_strategies as TS

# Trend-gate threshold grid
RET6_GRID = [0.003, 0.005, 0.007, 0.010]
MA20_GRID = [0.000, 0.002, 0.003, 0.005]

# Per-model approved gate config: (trend, group_weak)
MODEL_VARIANT = {
    "global6":   (True, True),
    "per_group": (True, False),
    "per_stock": (True, False),
}


def main() -> int:
    print("[1/2] building dataset (read-only)...", file=sys.stderr)
    feats, groups = TS._build_dataset()
    dates = sorted(feats["date"].unique())
    print(f"      rows={len(feats):,}  symbols={feats['symbol'].nunique()}  days={len(dates)}",
          file=sys.stderr)

    # Compute predictions ONCE per strategy (the expensive part)
    preds = {}
    for strat in ("global6", "per_group", "per_stock"):
        print(f"[2/2] walk-forward preds: {strat} ...", file=sys.stderr)
        p = BG._fold_preds(strat, feats, groups, dates)
        preds[strat] = p
        print(f"      {strat}: rows={len(p):,}", file=sys.stderr)

    for strat in ("global6", "per_group", "per_stock"):
        trend, gw = MODEL_VARIANT[strat]
        p = preds[strat]
        # baseline (no gate) for reference
        base = TS.realistic_pnl(BG._apply_gates(p, False, False), f"{strat}/baseline")
        print("\n" + "=" * 92)
        print(f"{strat}   variant: trend={trend} group_weak={gw}")
        print(f"  baseline (no gate): fills={base['fills']:>4}  win%={base['win_rate']:>5.1f}  "
              f"PnL={base['total_pnl']:>+9,}  PF={base.get('pf',0):>4.2f}  MDD={base['mdd']:>+9,}")
        print("=" * 92)
        hdr = (f"{'ret6':>6} {'ma20':>6} | {'fills':>5} {'win%':>6} "
               f"{'total NTD':>11} {'PF':>5} {'MDD':>9}")
        print(hdr)
        print("-" * 92)
        best = None
        for ret6 in RET6_GRID:
            for ma20 in MA20_GRID:
                BG.TREND_RET6 = ret6
                BG.TREND_MA20 = ma20
                pv = BG._apply_gates(p, trend, gw)
                s = TS.realistic_pnl(pv, f"{strat}")
                marker = ""
                if best is None or s["total_pnl"] > best[0]:
                    best = (s["total_pnl"], ret6, ma20, s)
                print(f"{ret6:>6.3f} {ma20:>6.3f} | {s['fills']:>5} {s['win_rate']:>6.1f} "
                      f"{s['total_pnl']:>+11,} {s.get('pf',0):>5.2f} {s['mdd']:>+9,}{marker}")
        b = best
        print("-" * 92)
        print(f"  >>> best PnL: ret6={b[1]} ma20={b[2]}  PnL={b[0]:>+,}  "
              f"win%={b[3]['win_rate']}  PF={b[3].get('pf',0)}  MDD={b[3]['mdd']:>+,}")
    print("\n" + "=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
