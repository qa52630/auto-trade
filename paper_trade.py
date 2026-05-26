"""Paper trading: EOD replay of today's signals through the live strategy.

Reads today's 1-min OHLCV CSVs (taishin_/masterlink_ format), reuses the
training pipeline to build 5-min features, loads `artifacts/model.joblib`,
generates short-only signals (proba ≤ 0.35) and simulates fills under the
realistic constraints (SL=2.5%, TP=3%, fee=0.225%, 60w daily budget,
1 lot/trade, -10k daily breaker).

Output (獨立於 live realtime 的 data/paper_trades/,避免覆蓋當日 live 資料):
  data/paper_backtest/YYYY-MM-DD.csv  — every signal with sim-fill outcome
  data/paper_backtest/summary.csv     — one row per day (appended idempotently)

Run:
  python3 paper_trade.py                  # today
  python3 paper_trade.py 2026-05-22       # specific date
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train as T

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
# 離線回測輸出獨立目錄,避免覆蓋 live realtime 寫的 data/paper_trades/{day}.csv
OUT_DIR = ROOT / "data" / "paper_backtest"
SUMMARY_CSV = OUT_DIR / "summary.csv"
BLACKLIST_DIR = ROOT / "data" / "blacklist"

# Strategy params
SHORT_SL = 0.025
SHORT_TP = 0.030
# Fee: 手續費 0.1425% × 0.6 折 × 2 邊 × 0.5 月退 (= 0.0855%) + 當沖證交稅 0.15% = 0.2355%
COST_RATE = 0.002355
PROBA_SHORT_THRESHOLD = 0.35
SHARES_PER_LOT = 1000
DAILY_BUDGET = 800_000
DAILY_LOSS_CAP = 10_000
EOD_HHMM = (13, 25)
# Entry filters
MAX_INTRADAY_PCT_FOR_SHORT = 0.07     # 漲幅 ≥ 7% 不放空 (避追高放空)

LIQUID_SYMBOLS = {"6770", "2337", "1802", "2408", "3481", "1815"}
HOLD_BARS = 6
THRESHOLD = 0.005
TZ = timezone(timedelta(hours=8))


def _load_blacklist(day: str) -> tuple[set[str], dict]:
    """Return (no_short_sell_set, info_dict). Combines 處置 + 不可現沖.
    Missing file → empty set + warning info."""
    p = BLACKLIST_DIR / f"{day}.json"
    if not p.exists():
        return set(), {"loaded": False, "path": str(p), "reason": "missing"}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return set(), {"loaded": False, "path": str(p), "reason": "bad_json"}
    # Schema v2 (Fugle): "no_short_sell" pre-computed
    if "no_short_sell" in data:
        return (
            set(str(c) for c in data["no_short_sell"]),
            {"loaded": True, "path": str(p),
             "disposition_n": len(data.get("disposition", [])),
             "no_day_trade_n": len(data.get("no_day_trade", [])),
             "no_short_sell_n": len(data.get("no_short_sell", []))},
        )
    # Schema v1 fallback (TWSE-only): disposition + suspended_short
    blocked = set(str(c) for c in data.get("disposition", [])) \
            | set(str(c) for c in data.get("suspended_short", []))
    return blocked, {"loaded": True, "path": str(p), "no_short_sell_n": len(blocked)}


def _load_1m_for_day(day: str, symbol: str) -> pd.DataFrame | None:
    base = ROOT / "data" / day[:7]
    for prefix in ("masterlink", "taishin"):
        p = base / f"{prefix}_{day}_{symbol}_1m.csv"
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
        return df[cols].sort_values("dt").reset_index(drop=True)
    return None


def _generate_signals_for_day(day: str) -> pd.DataFrame:
    """Run the full feature/prediction pipeline restricted to `day` predictions.
    Requires enough history before `day` for rolling features (~20 days)."""
    print(f"[1/3] loading all 1-min CSVs for feature context...", file=sys.stderr)
    df1 = T.load_all()
    if day not in set(df1["date"]):
        raise SystemExit(f"no 1-min data for {day} in data/")

    df5 = T.resample_5m(df1)
    feats = T.build_features(df5)
    feats = T.add_cross_sectional(feats, ROOT / "stocks.json")
    feats = feats[feats["symbol"].isin(LIQUID_SYMBOLS)]
    # Labels (fwd_ret) not needed for paper trade but keeping pipeline identical
    feats = T.label_next(feats, threshold=THRESHOLD, hold_bars=HOLD_BARS)
    feats = feats.dropna(subset=T.FEATURE_COLS)
    today = feats[feats["date"] == day].copy()
    if today.empty:
        raise SystemExit(f"no feature rows for {day} after pipeline")
    # liquidity gate (entry-bar only; we cannot peek at exit-bar liquidity live)
    today = today[(today["volume"] >= 1000) & (today["high"] != today["low"])]
    print(f"      feature rows for {day}: {len(today)} across {today['symbol'].nunique()} symbols", file=sys.stderr)

    print(f"[2/3] loading model {MODEL_PATH.name}...", file=sys.stderr)
    if not MODEL_PATH.exists():
        raise SystemExit("model.joblib not found — run train.py first")
    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(today[T.FEATURE_COLS].values)[:, 1]
    today["proba"] = proba
    today["conf"] = 1 - proba
    today["triggered"] = proba <= PROBA_SHORT_THRESHOLD
    return today


def _simulate_fills(day: str, signals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk signals chronologically, simulate next-bar entry + intraday SL/TP."""
    print(f"[3/3] simulating fills (SL={SHORT_SL*100:.1f}% TP={SHORT_TP*100:.1f}% "
          f"fee={COST_RATE*100:.3f}% budget={DAILY_BUDGET:,} breaker=-{DAILY_LOSS_CAP:,})", file=sys.stderr)

    no_short_sell, bl_info = _load_blacklist(day)
    if bl_info["loaded"]:
        print(f"      blacklist: 不可放空清單 {bl_info.get('no_short_sell_n', 0)} 檔 "
              f"(處置 {bl_info.get('disposition_n','?')} + 不可現沖 {bl_info.get('no_day_trade_n','?')})",
              file=sys.stderr)
    else:
        print(f"      ⚠️  blacklist file 缺失 ({bl_info['path']}) — 處置股/不可現沖過濾未啟用",
              file=sys.stderr)

    signals = signals.sort_values("datetime").reset_index(drop=True)
    signals["signal_dt"] = pd.to_datetime(signals["datetime"]).dt.tz_localize(None)
    m1_cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    skipped = defaultdict(int)
    budget_used = 0.0
    daily_pnl = 0.0
    breaker = False
    # #1 fix: track per-symbol last exit time. If a new signal would overlap
    # the previous still-open position for that symbol, skip it.
    held_until: dict[str, pd.Timestamp] = {}

    for _, s in signals.iterrows():
        sym = str(s["symbol"])
        row_base = {
            "datetime": s["datetime"], "symbol": sym, "close": float(s["close"]),
            "proba": round(float(s["proba"]), 4), "conf": round(float(s["conf"]), 3),
            "triggered": bool(s["triggered"]),
        }
        if not s["triggered"]:
            row_base["status"] = "no_signal"
            rows.append(row_base); continue
        if breaker:
            row_base["status"] = "skip_breaker"; skipped["breaker"] += 1
            rows.append(row_base); continue
        # #1 fix: this signal's entry is signal_dt + 5min; if the previous
        # position for this symbol hasn't exited by then → skip duplicate.
        sig_entry_dt = s["signal_dt"] + pd.Timedelta(minutes=5)
        if sym in held_until and sig_entry_dt < held_until[sym]:
            row_base["status"] = "skip_already_open"
            skipped["already_open"] += 1
            rows.append(row_base); continue
        # Entry filter: 不可放空 (處置 + 不可現沖,合併判定)
        if sym in no_short_sell:
            row_base["status"] = "skip_no_short_sell"; skipped["no_short_sell"] += 1
            rows.append(row_base); continue

        m1 = m1_cache.get(sym)
        if m1 is None:
            m1 = _load_1m_for_day(day, sym)
            m1_cache[sym] = m1
        if m1 is None:
            row_base["status"] = "skip_no1m"; skipped["no1m"] += 1
            rows.append(row_base); continue

        # Entry filter: 漲幅 ≥ 7% 不放空
        ref_price = None
        if "ref_price" in m1.columns:
            rp = m1["ref_price"].dropna()
            rp = rp[rp > 0]
            if len(rp):
                ref_price = float(rp.iloc[0])
        if ref_price and ref_price > 0:
            intraday_pct = (float(s["close"]) - ref_price) / ref_price
            row_base["intraday_pct"] = round(intraday_pct, 4)
            if intraday_pct >= MAX_INTRADAY_PCT_FOR_SHORT:
                row_base["status"] = "skip_pct_high"; skipped["pct_high"] += 1
                rows.append(row_base); continue

        entry_dt = s["signal_dt"] + pd.Timedelta(minutes=5)
        after = m1[m1["dt"] >= entry_dt]
        if after.empty:
            row_base["status"] = "skip_no_next_bar"; skipped["nobar"] += 1
            rows.append(row_base); continue
        entry_price = float(after.iloc[0]["open"])
        if entry_price <= 0:
            row_base["status"] = "skip_bad_price"; skipped["badpx"] += 1
            rows.append(row_base); continue

        notional = entry_price * SHARES_PER_LOT
        if budget_used + notional > DAILY_BUDGET:
            row_base["status"] = "skip_budget"; skipped["budget"] += 1
            rows.append(row_base); continue

        sl_px = entry_price * (1 + SHORT_SL)
        tp_px = entry_price * (1 - SHORT_TP)
        exit_price = exit_reason = None
        exit_detail = ""; exit_dt_ts = None
        for _, b in after.iloc[1:].iterrows():
            if (b["dt"].hour, b["dt"].minute) >= EOD_HHMM:
                exit_price, exit_reason = float(b["open"]), "eod"
                exit_dt_ts = b["dt"]
                exit_detail = f"13:25 EOD force-close at {b['dt'].time().isoformat()[:5]} open={b['open']:.2f}"
                break
            if b["high"] >= sl_px:
                exit_price, exit_reason = sl_px, "sl"; exit_dt_ts = b["dt"]
                exit_detail = f"high={b['high']:.2f} ≥ sl_px={sl_px:.4f} at {b['dt'].time().isoformat()[:5]}"
                break
            if b["low"] <= tp_px:
                exit_price, exit_reason = tp_px, "tp"; exit_dt_ts = b["dt"]
                exit_detail = f"low={b['low']:.2f} ≤ tp_px={tp_px:.4f} at {b['dt'].time().isoformat()[:5]}"
                break
        if exit_price is None:
            exit_price, exit_reason = float(after.iloc[-1]["close"]), "last"
            exit_dt_ts = after.iloc[-1]["dt"]
            exit_detail = f"end of data; closed at last bar {exit_dt_ts.time().isoformat()[:5]} close={exit_price:.2f}"

        gross_ret = (entry_price - exit_price) / entry_price
        net_ret = gross_ret - COST_RATE
        ntd_pnl = notional * net_ret
        budget_used += notional
        daily_pnl += ntd_pnl
        if daily_pnl <= -DAILY_LOSS_CAP:
            breaker = True

        try:
            holding_min = round((exit_dt_ts - after.iloc[0]["dt"]).total_seconds() / 60.0, 1)
        except Exception:
            holding_min = None

        row_base.update({
            "status": "filled",
            "entry_dt": after.iloc[0]["dt"].isoformat(),
            "entry_price": round(entry_price, 2),
            "entry_reason": f"short proba={s['proba']:.3f} ≤ {PROBA_SHORT_THRESHOLD}; filters passed",
            "exit_dt": exit_dt_ts.isoformat() if exit_dt_ts is not None else None,
            "exit_price": round(exit_price, 3),
            "exit_reason": exit_reason,
            "exit_detail": exit_detail,
            "holding_min": holding_min,
            "lots": 1,
            "notional": round(notional, 0),
            "gross_ret": round(gross_ret, 5),
            "net_ret": round(net_ret, 5),
            "ntd_pnl": round(ntd_pnl, 2),
        })
        rows.append(row_base)
        # #1 fix: remember this position's exit time so future overlapping signals get skipped
        if exit_dt_ts is not None:
            held_until[sym] = exit_dt_ts

    out = pd.DataFrame(rows)
    stats = {
        "filled": int((out["status"] == "filled").sum()),
        "skipped_budget": skipped["budget"],
        "skipped_breaker": skipped["breaker"],
        "skipped_already_open": skipped["already_open"],
        "skipped_no_short_sell": skipped["no_short_sell"],
        "skipped_pct_high": skipped["pct_high"],
        "skipped_other": skipped["no1m"] + skipped["nobar"] + skipped["badpx"],
        "breaker_tripped": breaker,
        "budget_used": int(round(budget_used)),
        "daily_pnl": int(round(daily_pnl)),
        "blacklist_loaded": bl_info["loaded"],
    }
    return out, stats


def _write_summary(day: str, stats: dict, trades: pd.DataFrame) -> None:
    filled = trades[trades["status"] == "filled"]
    if len(filled):
        win_rate = float((filled["ntd_pnl"] > 0).mean())
        tp_count = int((filled["exit_reason"] == "tp").sum())
        sl_count = int((filled["exit_reason"] == "sl").sum())
    else:
        win_rate = 0.0; tp_count = sl_count = 0

    row = {
        "date": day,
        "signals_total": int(len(trades)),
        "signals_triggered": int(trades["triggered"].sum()),
        "filled": stats["filled"],
        "skipped_budget": stats["skipped_budget"],
        "skipped_breaker": stats["skipped_breaker"],
        "win_rate": round(win_rate, 3),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "ntd_pnl": stats["daily_pnl"],
        "budget_used": stats["budget_used"],
        "breaker_tripped": stats["breaker_tripped"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["date"] != day]   # idempotent: replace existing row for this day
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("day", nargs="?", default=None, help="YYYY-MM-DD; default today TPE")
    args = p.parse_args()
    day = args.day or datetime.now(TZ).date().isoformat()
    print(f"=== paper_trade for {day} ===", file=sys.stderr)

    signals = _generate_signals_for_day(day)
    trades, stats = _simulate_fills(day, signals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{day}.csv"
    trades.to_csv(out_path, index=False)
    _write_summary(day, stats, trades)

    print(f"\n=== {day} 結果 ===")
    print(f"訊號 (全部 / 觸發 / 實際成交): {len(trades)} / {int(trades['triggered'].sum())} / {stats['filled']}")
    print(f"跳過: 預算={stats['skipped_budget']} 熔斷後={stats['skipped_breaker']} "
          f"不可放空={stats['skipped_no_short_sell']} 漲幅≥7%={stats['skipped_pct_high']} "
          f"其他={stats['skipped_other']}")
    if not stats.get("blacklist_loaded"):
        print(f"⚠️  blacklist 未載入 — 處置/不可現沖過濾未啟用")
    if stats["filled"]:
        filled = trades[trades["status"] == "filled"]
        wr = (filled["ntd_pnl"] > 0).mean()
        print(f"成交勝率: {wr:.1%}")
        print(f"日 NTD 損益: {stats['daily_pnl']:+,}  (本金 30 萬 = {stats['daily_pnl']/300000*100:+.2f}%)")
        print(f"預算使用率: {stats['budget_used']/DAILY_BUDGET:.0%}")
        if stats["breaker_tripped"]:
            print(f"⚠️  日內熔斷觸發 (daily PnL ≤ -{DAILY_LOSS_CAP:,})")
    else:
        print("無成交")
    print(f"\n明細 → {out_path}")
    print(f"摘要 → {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
