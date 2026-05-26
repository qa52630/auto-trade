"""Daily PnL comparison: auto-trade (3 strategies) vs stock-python (live)

Reads both projects' trade outputs and produces a side-by-side daily table.

Sources:
  auto-trade:   data/paper_trades/{day}.csv (global6)
                data/paper_trades/{day}_per_group.csv
                data/paper_trades/{day}_per_stock.csv
  stock-python: /Users/ben/stock-python/live_results/{YYYY-MM}/live_trades_{day}.csv

Output:
  - Console table by day
  - data/comparison/{YYYY-MM-DD}_compare.json (raw side-by-side)
  - data/comparison/summary.csv (running monthly totals)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Auto-detect AT_ROOT: this file lives at AT_ROOT root, so use its parent dir.
AT_ROOT = Path(__file__).parent
SP_ROOT = Path("/Users/ben/stock-python")
OUT_DIR = AT_ROOT / "data" / "comparison"
TZ = timezone(timedelta(hours=8))

STRATEGIES = ["global6", "per_group", "per_stock"]


def _at_pnl_for_day(day: str) -> dict[str, dict]:
    """Read auto-trade trades for `day` across all 3 strategies."""
    result = {}
    for strat in STRATEGIES:
        suffix = "" if strat == "global6" else f"_{strat}"
        p = AT_ROOT / "data" / "paper_trades" / f"{day}{suffix}.csv"
        if not p.exists():
            result[strat] = {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": "no_file"}
            continue
        try:
            df = pd.read_csv(p, dtype={"symbol": str})
        except Exception as e:
            result[strat] = {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": f"err: {e}"}
            continue
        if "status" not in df.columns:
            result[strat] = {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": "no_status_col"}
            continue
        filled = df[df["status"].isin(["filled", "closed"])].copy()
        if filled.empty:
            result[strat] = {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": "no_fills"}
            continue
        wins = int((filled["ntd_pnl"] > 0).sum())
        result[strat] = {
            "trades": int(len(filled)),
            "wins": wins,
            "ntd_pnl": int(round(filled["ntd_pnl"].sum())),
            "win_rate": round(wins / len(filled), 3),
            "status": "ok",
        }
    return result


def _sp_pnl_for_day(day: str) -> dict:
    """Read stock-python live trades for `day`."""
    month = day[:7]
    p = SP_ROOT / "live_results" / month / f"live_trades_{day}.csv"
    if not p.exists():
        return {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": "no_file"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": f"err: {e}"}
    if df.empty:
        return {"trades": 0, "wins": 0, "ntd_pnl": 0, "status": "empty"}
    # profit column already nets commission + tax
    pnl = df["profit"].sum()
    wins = int((df["profit"] > 0).sum())
    return {
        "trades": int(len(df)),
        "wins": wins,
        "ntd_pnl": int(round(pnl)),
        "win_rate": round(wins / len(df), 3) if len(df) else 0,
        "shorts": int((df["direction"] == "short").sum()),
        "longs": int((df["direction"] == "long").sum()),
        "by_exit": df.groupby("exit_reason")["profit"].agg(["count", "sum"]).round(0).to_dict(),
        "status": "ok",
    }


def _list_days(start: str, end: str) -> list[str]:
    """Return list of YYYY-MM-DD between start and end inclusive, weekdays only."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def compare(days: list[str]) -> list[dict]:
    rows = []
    for day in days:
        at = _at_pnl_for_day(day)
        sp = _sp_pnl_for_day(day)
        # Skip days with no data on either side
        if all(at[s]["trades"] == 0 for s in STRATEGIES) and sp["trades"] == 0:
            continue
        rows.append({"day": day, "auto_trade": at, "stock_python": sp})
    return rows


def upsert_daily_log(rows: list[dict]) -> Path:
    """Append/update each day's comparison into data/comparison/daily_log.csv.
    Existing rows for the same date are replaced (idempotent re-runs)."""
    log_path = OUT_DIR / "daily_log.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "date",
        "at_g6_trades", "at_g6_winrate", "at_g6_pnl",
        "at_pg_trades", "at_pg_winrate", "at_pg_pnl",
        "at_ps_trades", "at_ps_winrate", "at_ps_pnl",
        "sp_trades", "sp_winrate", "sp_pnl",
        "best_at_pnl", "delta_vs_sp", "verdict",
        "logged_at",
    ]

    # Load existing (if any) keyed by date
    existing: dict[str, dict] = {}
    if log_path.exists():
        with log_path.open() as f:
            for r in csv.DictReader(f):
                existing[r["date"]] = r

    now = datetime.now(TZ).isoformat(timespec="seconds")
    for r in rows:
        day = r["day"]
        at = r["auto_trade"]; sp = r["stock_python"]
        g6, pg, ps = at["global6"], at["per_group"], at["per_stock"]
        at_pnls = [s["ntd_pnl"] for s in (g6, pg, ps) if s["trades"]]
        best_at = max(at_pnls) if at_pnls else 0
        delta = best_at - sp["ntd_pnl"]
        if best_at > 0 and sp["ntd_pnl"] < 0:
            verdict = "at_win_sp_lose"
        elif best_at > sp["ntd_pnl"]:
            verdict = "at_better"
        elif best_at < 0 and sp["ntd_pnl"] < 0:
            verdict = "both_lose"
        else:
            verdict = "at_worse"
        existing[day] = {
            "date": day,
            "at_g6_trades": g6["trades"], "at_g6_winrate": g6.get("win_rate", 0), "at_g6_pnl": g6["ntd_pnl"],
            "at_pg_trades": pg["trades"], "at_pg_winrate": pg.get("win_rate", 0), "at_pg_pnl": pg["ntd_pnl"],
            "at_ps_trades": ps["trades"], "at_ps_winrate": ps.get("win_rate", 0), "at_ps_pnl": ps["ntd_pnl"],
            "sp_trades": sp["trades"], "sp_winrate": sp.get("win_rate", 0), "sp_pnl": sp["ntd_pnl"],
            "best_at_pnl": best_at, "delta_vs_sp": delta, "verdict": verdict,
            "logged_at": now,
        }

    # Write back sorted by date
    out_rows = sorted(existing.values(), key=lambda r: r["date"])
    tmp = log_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    tmp.replace(log_path)
    return log_path


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no data)")
        return

    print(f"\n{'day':<11} {'g6 fills/win/PnL':>20}  {'pg fills/win/PnL':>20}  "
          f"{'ps fills/win/PnL':>20} | {'SP fills/win/PnL':>20}  {'差距':>9}")
    print("-" * 130)

    sums = {"global6": 0, "per_group": 0, "per_stock": 0, "sp": 0}
    for r in rows:
        at = r["auto_trade"]; sp = r["stock_python"]
        cells = []
        for strat in STRATEGIES:
            s = at[strat]
            if s["trades"]:
                cells.append(f"{s['trades']:>2}/{int(s['win_rate']*100):>3}%/{s['ntd_pnl']:>+7,}")
                sums[strat] += s["ntd_pnl"]
            else:
                cells.append(f"{'(no data)':>20}")
        if sp["trades"]:
            sp_cell = f"{sp['trades']:>2}/{int(sp['win_rate']*100):>3}%/{sp['ntd_pnl']:>+7,}"
            sums["sp"] += sp["ntd_pnl"]
        else:
            sp_cell = "(no data)"
        at_pnls = [at[s]["ntd_pnl"] for s in STRATEGIES if at[s]["trades"]]
        best_at = max(at_pnls) if at_pnls else 0
        delta = best_at - sp["ntd_pnl"] if sp["trades"] else 0
        print(f"{r['day']:<11} {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20} | "
              f"{sp_cell:>20}  {delta:>+9,}")

    print("-" * 130)
    print(f"{'累計':<11} "
          f"{sums['global6']:>+20,}  {sums['per_group']:>+20,}  {sums['per_stock']:>+20,} | "
          f"{sums['sp']:>+20,}  {max(sums['global6'],sums['per_group'],sums['per_stock'])-sums['sp']:>+9,}")
    print()

    # Verdict
    best_at = max(sums["global6"], sums["per_group"], sums["per_stock"])
    sp_v = sums["sp"]
    if best_at > 0 and sp_v < 0:
        verdict = "✅ auto-trade 賺、stock-python 賠 → 我們策略有相對優勢"
    elif best_at > sp_v:
        verdict = f"⚠️ auto-trade 比 stock-python 多賺 {best_at - sp_v:+,} — 都正/都微負,差距觀察"
    elif best_at < 0 and sp_v < 0:
        verdict = "❌ 兩邊都虧 → universe 問題,該換標的或停手"
    else:
        verdict = "🔴 auto-trade 不如 stock-python — 白工警訊"
    print(f"判讀: {verdict}")


def main() -> int:
    today = datetime.now(TZ).date().isoformat()
    default_start = (datetime.now(TZ).date() - timedelta(days=30)).isoformat()
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=default_start, help=f"YYYY-MM-DD (default: 30 days ago)")
    p.add_argument("--end", default=today, help="YYYY-MM-DD (default: today)")
    args = p.parse_args()

    days = _list_days(args.start, args.end)
    print(f"comparing {len(days)} weekdays: {args.start} ~ {args.end}", file=sys.stderr)

    rows = compare(days)
    print_table(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{args.start}_to_{args.end}.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    log_path = upsert_daily_log(rows)
    print(f"\n明細 → {out_json}", file=sys.stderr)
    print(f"累積 log → {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
