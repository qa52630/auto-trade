"""Simple web dashboard for auto-trade backtest results.

Run:   python3 dashboard.py
Open:  http://localhost:5050
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).parent
TRADES_CSV = ROOT / "artifacts" / "trades_full_oos.csv"
PREDS_CSV = ROOT / "artifacts" / "wf_preds_extended.csv"
PAPER_DIR = ROOT / "data" / "paper_trades"
PAPER_SUMMARY = PAPER_DIR / "summary.csv"
UNIVERSE_JSON = ROOT / "data" / "universe.json"

app = Flask(__name__)


def _load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_CSV, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["month"] = df["date"].str.slice(0, 7)
    df["win"] = df["ntd_pnl"] > 0
    df = df.sort_values("signal_dt").reset_index(drop=True)
    df["cum_pnl"] = df["ntd_pnl"].cumsum()
    return df


def _load_signals() -> pd.DataFrame:
    df = pd.read_csv(PREDS_CSV, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["month"] = df["date"].str.slice(0, 7)
    df["conf"] = (1 - df["proba"]).round(3)
    df["triggered"] = df["position"] != 0
    return df


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/summary")
def api_summary():
    month = request.args.get("month")
    t = _load_trades()
    if month and month != "all":
        t = t[t["month"] == month]
    if len(t) == 0:
        return jsonify({"empty": True})

    cum = t["ntd_pnl"].cumsum()
    peak = cum.cummax()
    mdd = float((cum - peak).min())
    wins = t.loc[t["win"], "ntd_pnl"]
    losses = t.loc[~t["win"], "ntd_pnl"]
    pf = float(abs(wins.sum() / losses.sum())) if len(losses) and losses.sum() != 0 else 0.0
    return jsonify({
        "trades": int(len(t)),
        "win_rate": round(float(t["win"].mean()) * 100, 1),
        "total_pnl": int(round(t["ntd_pnl"].sum())),
        "avg_pnl": int(round(t["ntd_pnl"].mean())),
        "max_win": int(round(t["ntd_pnl"].max())),
        "max_loss": int(round(t["ntd_pnl"].min())),
        "mdd": int(round(mdd)),
        "pf": round(pf, 2),
        "sl_count": int((t["reason"] == "sl").sum()),
        "tp_count": int((t["reason"] == "tp").sum()),
        "eod_count": int((t["reason"] == "eod").sum()),
    })


@app.route("/api/monthly")
def api_monthly():
    t = _load_trades()
    g = t.groupby("month").agg(
        trades=("ntd_pnl", "size"),
        win_rate=("win", "mean"),
        ntd_pnl=("ntd_pnl", "sum"),
    ).reset_index()
    g["win_rate"] = (g["win_rate"] * 100).round(1)
    g["ntd_pnl"] = g["ntd_pnl"].round(0).astype(int)
    return jsonify(g.to_dict(orient="records"))


@app.route("/api/equity")
def api_equity():
    month = request.args.get("month")
    t = _load_trades()
    if month and month != "all":
        t = t[t["month"] == month].copy()
        t["cum_pnl"] = t["ntd_pnl"].cumsum()
    return jsonify([
        {"signal_dt": r["signal_dt"], "cum_pnl": float(round(r["cum_pnl"], 0))}
        for _, r in t.iterrows()
    ])


@app.route("/api/trades")
def api_trades():
    month = request.args.get("month")
    t = _load_trades()
    if month and month != "all":
        t = t[t["month"] == month]
    return jsonify([
        {
            "signal_dt": r["signal_dt"],
            "symbol": r["symbol"],
            "entry": round(float(r["entry"]), 2),
            "exit": round(float(r["exit"]), 3),
            "reason": r["reason"],
            "conf": round(float(r["conf"]), 3),
            "ntd_pnl": int(round(r["ntd_pnl"])),
            "fold": int(r["fold"]),
        }
        for _, r in t.iterrows()
    ])


from flask import request as _flask_request


@app.route("/api/universe", methods=["GET", "POST"])
def api_universe():
    default = {"active_symbols": ["6770", "2337", "1802", "2408", "3481", "1815"],
               "candidates": ["6770", "2337", "1802", "2408", "3481", "1815"]}
    if request.method == "POST":
        body = request.get_json(force=True) or {}
        new_active = body.get("active_symbols")
        if not isinstance(new_active, list) or not all(isinstance(s, str) for s in new_active):
            return jsonify({"error": "active_symbols must be a list of strings"}), 400
        # Load current to preserve candidates
        cur = json.loads(UNIVERSE_JSON.read_text()) if UNIVERSE_JSON.exists() else default
        # Only allow symbols that are in candidates list
        candidates = set(cur.get("candidates", default["candidates"]))
        active = [s for s in new_active if s in candidates]
        cur["active_symbols"] = active
        from datetime import timezone as _tz, timedelta as _td
        cur["last_updated"] = datetime.now(_tz(_td(hours=8))).isoformat()
        UNIVERSE_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
        return jsonify(cur)
    if not UNIVERSE_JSON.exists():
        return jsonify(default)
    return jsonify(json.loads(UNIVERSE_JSON.read_text()))


@app.route("/api/universe_candidates", methods=["POST"])
def api_universe_candidates():
    body = request.get_json(force=True) or {}
    sym = (body.get("symbol") or "").strip()
    if not sym.isdigit() or not (4 <= len(sym) <= 6):
        return jsonify({"error": "symbol must be 4-6 digits"}), 400
    cur = json.loads(UNIVERSE_JSON.read_text()) if UNIVERSE_JSON.exists() else {
        "active_symbols": [], "candidates": []
    }
    cands = cur.get("candidates", [])
    if sym not in cands:
        cands.append(sym)
        cur["candidates"] = cands
    UNIVERSE_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    return jsonify(cur)


@app.route("/api/today")
def api_today():
    """Aggregate today's state across all strategies + watchlist info."""
    from datetime import timezone as _tz, timedelta as _td
    tz = _tz(_td(hours=8))
    today = datetime.now(tz).date().isoformat()
    day = request.args.get("day", today)

    # Universe & blacklist
    universe = {"active_symbols": [], "candidates": []}
    if UNIVERSE_JSON.exists():
        universe = json.loads(UNIVERSE_JSON.read_text())
    blacklist_path = ROOT / "data" / "blacklist" / f"{day}.json"
    tickers = {}
    no_short = set()
    if blacklist_path.exists():
        try:
            bl = json.loads(blacklist_path.read_text())
            tickers = bl.get("tickers", {})
            no_short = set(bl.get("no_short_sell", []))
        except json.JSONDecodeError:
            pass
    watchlist = []
    for sym in universe.get("active_symbols", []):
        t = tickers.get(sym, {})
        watchlist.append({
            "symbol": sym,
            "name": t.get("name", ""),
            "ref_price": t.get("referencePrice"),
            "limit_up": t.get("limitUpPrice"),
            "limit_down": t.get("limitDownPrice"),
            "blocked": sym in no_short,
            "is_disposition": bool(t.get("isDisposition")),
            "is_attention": bool(t.get("isAttention")),
            "can_day_trade": bool(t.get("canDayTrade")) if t else None,
        })

    # Per-strategy state
    strategies = {}
    for strat in ("global6", "per_group", "per_stock"):
        suffix = "" if strat == "global6" else f"_{strat}"
        state_p = PAPER_DIR / f"state_{day}{suffix}.json"
        if not state_p.exists():
            strategies[strat] = {"open": [], "closed": [], "daily_pnl": 0,
                                 "budget_used": 0, "breaker_tripped": False}
            continue
        try:
            s = json.loads(state_p.read_text())
        except json.JSONDecodeError:
            continue
        strategies[strat] = {
            "open": s.get("open_positions", []),
            "closed": s.get("closed_trades", []),
            "daily_pnl": s.get("daily_pnl", 0),
            "budget_used": s.get("budget_used", 0),
            "breaker_tripped": s.get("breaker_tripped", False),
        }

    # TAIEX regime gate status (importing lazily so dashboard works even if
    # live_paper module shape changes)
    taiex_gate = {"block": False, "trend": None, "gap": None, "reason": "unavailable"}
    try:
        from live_paper import taiex_regime_gate
        block, info = taiex_regime_gate(day)
        taiex_gate = {
            "block": bool(block),
            "trend": info.get("trend"),
            "gap": info.get("gap"),
            "reason": info.get("reason", ""),
        }
    except Exception as e:
        taiex_gate["reason"] = f"error: {type(e).__name__}"

    return jsonify({
        "day": day,
        "watchlist": watchlist,
        "strategies": strategies,
        "taiex_gate": taiex_gate,
    })


@app.route("/api/history_entries")
def api_history_entries():
    """Flat list of all historical filled/closed trades across files."""
    if not PAPER_DIR.exists():
        return jsonify([])
    rows = []
    for f in sorted(PAPER_DIR.glob("*.csv"), reverse=True):
        name = f.stem
        if name == "summary":
            continue
        if "_" in name and len(name.split("_")[0]) == 10:
            day, strat = name.split("_", 1)
        else:
            day, strat = name, "global6"
        try:
            df = pd.read_csv(f, dtype={"symbol": str})
        except Exception:
            continue
        if "status" not in df.columns:
            continue
        filled = df[df["status"].isin(["filled", "closed"])].copy()
        if filled.empty:
            continue
        for _, t in filled.iterrows():
            rows.append({
                "day": day, "strategy": strat,
                "datetime": str(t.get("datetime", ""))[:16],
                "entry_dt": str(t.get("entry_dt", ""))[11:16] if t.get("entry_dt") else "",
                "exit_dt": str(t.get("exit_dt", ""))[11:16] if t.get("exit_dt") else "",
                "symbol": str(t.get("symbol", "")),
                "entry_price": float(t.get("entry_price", 0) or 0),
                "exit_price": float(t.get("exit_price", 0) or 0),
                "exit_reason": str(t.get("exit_reason", "")),
                "holding_min": float(t.get("holding_min", 0) or 0),
                "ntd_pnl": int(round(float(t.get("ntd_pnl", 0) or 0))),
            })
    # Newest first
    rows.sort(key=lambda r: (r["day"], r["datetime"]), reverse=True)
    return jsonify(rows[:500])  # cap


@app.route("/api/paper_summary")
def api_paper_summary():
    """Per-day P&L summary, derived live from the daily trade CSVs.

    Sums across all 3 strategies (global6 + per_group + per_stock) so the
    headline bar reflects the real account total. No longer depends on the
    legacy summary.csv (written only by the 14:30 paper_trade cron, which is
    fragile and overwrites live data) — this auto-updates every EOD.
    """
    if not PAPER_DIR.exists():
        return jsonify([])
    by_day: dict[str, dict] = {}
    for f in PAPER_DIR.glob("*.csv"):
        name = f.stem
        if name.startswith("state_") or name == "summary":
            continue
        day = name.split("_", 1)[0] if (len(name.split("_")[0]) == 10) else name
        try:
            df = pd.read_csv(f, dtype={"symbol": str})
        except Exception:
            continue
        agg = by_day.setdefault(day, {
            "date": day, "filled": 0, "ntd_pnl": 0.0, "_wins": 0,
            "tp_count": 0, "sl_count": 0, "breaker_tripped": False,
        })
        if "status" in df.columns:
            filled = df[df["status"].isin(["filled", "closed"])]
        else:
            filled = df
        if filled.empty or "ntd_pnl" not in filled.columns:
            continue
        pnl = pd.to_numeric(filled["ntd_pnl"], errors="coerce").fillna(0)
        agg["filled"] += int(len(filled))
        agg["ntd_pnl"] += float(pnl.sum())
        agg["_wins"] += int((pnl > 0).sum())
        if "exit_reason" in filled.columns:
            er = filled["exit_reason"].astype(str).str.lower()
            agg["tp_count"] += int(er.str.contains("tp").sum())
            agg["sl_count"] += int(er.str.contains("sl").sum())
    rows = []
    for day, a in by_day.items():
        n = a["filled"]
        rows.append({
            "date": day,
            "filled": n,
            "ntd_pnl": int(round(a["ntd_pnl"])),
            "win_rate": round(a["_wins"] / n, 3) if n else 0,
            "tp_count": a["tp_count"],
            "sl_count": a["sl_count"],
            "breaker_tripped": a["breaker_tripped"],
        })
    rows.sort(key=lambda r: r["date"])
    return jsonify(rows)


@app.route("/api/paper_trades")
def api_paper_trades():
    day = request.args.get("day")
    strategy = request.args.get("strategy", "global6")
    if not day:
        return jsonify([])
    suffix = "" if strategy == "global6" else f"_{strategy}"
    p = PAPER_DIR / f"{day}{suffix}.csv"
    if not p.exists():
        return jsonify([])
    df = pd.read_csv(p, dtype={"symbol": str})
    # Keep rows that are either triggered signals or actual position events.
    keep = df["triggered"].fillna(False) == True
    if "status" in df.columns:
        keep = keep | df["status"].isin(["filled", "closed", "open"])
    df = df[keep].fillna("")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/paper_strategies")
def api_paper_strategies():
    """List which strategies have data for which days."""
    if not PAPER_DIR.exists():
        return jsonify({"days": [], "strategies": []})
    days = set(); strats = set()
    for f in PAPER_DIR.glob("*.csv"):
        name = f.stem
        if name.startswith("state_") or name == "summary":
            continue
        if "_" in name and len(name.split("_")[0]) == 10:   # YYYY-MM-DD_xxx
            day, strat = name.split("_", 1)
        else:
            day, strat = name, "global6"
        days.add(day); strats.add(strat)
    return jsonify({"days": sorted(days, reverse=True), "strategies": sorted(strats)})


@app.route("/api/paper_compare")
def api_paper_compare():
    """Return per-day per-strategy daily P&L summary for stacked comparison."""
    if not PAPER_DIR.exists():
        return jsonify([])
    by_day_strat: dict[tuple[str, str], dict] = {}
    for f in PAPER_DIR.glob("*.csv"):
        name = f.stem
        if name.startswith("state_") or name == "summary":
            continue
        if "_" in name and len(name.split("_")[0]) == 10:
            day, strat = name.split("_", 1)
        else:
            day, strat = name, "global6"
        try:
            df = pd.read_csv(f, dtype={"symbol": str})
        except Exception:
            continue
        # Treat both EOD ("filled") and live ("closed") completed trades as fills
        if "status" in df.columns:
            filled = df[df["status"].isin(["filled", "closed"])]
        else:
            filled = df
        if filled.empty or "ntd_pnl" not in filled.columns:
            by_day_strat[(day, strat)] = {"day": day, "strategy": strat, "fills": 0,
                                          "ntd_pnl": 0, "win_rate": 0}
            continue
        wins = int((filled["ntd_pnl"] > 0).sum())
        by_day_strat[(day, strat)] = {
            "day": day, "strategy": strat,
            "fills": int(len(filled)),
            "ntd_pnl": int(round(filled["ntd_pnl"].sum())),
            "win_rate": round(wins / len(filled), 3) if len(filled) else 0,
        }
    return jsonify(sorted(by_day_strat.values(), key=lambda r: (r["day"], r["strategy"])))


@app.route("/api/signals")
def api_signals():
    """Raw model signals (incl. those filtered out by position threshold or budget)."""
    month = request.args.get("month")
    only_triggered = request.args.get("triggered") == "1"
    s = _load_signals()
    if month and month != "all":
        s = s[s["month"] == month]
    if only_triggered:
        s = s[s["triggered"]]
    s = s.sort_values("datetime", ascending=False).head(1000)
    return jsonify([
        {
            "datetime": r["datetime"],
            "symbol": r["symbol"],
            "close": float(r["close"]),
            "proba": round(float(r["proba"]), 4),
            "conf": float(r["conf"]),
            "fwd_ret": round(float(r["fwd_ret"]), 5),
            "triggered": bool(r["triggered"]),
            "fold": int(r["fold"]),
        }
        for _, r in s.iterrows()
    ])


PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>Auto-Trade Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 20px; background: #f5f5f7; color: #1d1d1f; }
  h1 { margin: 0 0 8px; font-size: 22px; }
  .subtitle { color: #86868b; font-size: 13px; margin-bottom: 16px; }
  .filter { margin-bottom: 16px; }
  .filter select { padding: 6px 10px; border-radius: 6px; border: 1px solid #d2d2d7; background: white; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: white; padding: 14px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .card .label { font-size: 11px; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .pos { color: #2db849; }
  .neg { color: #d8302a; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel { background: white; padding: 16px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .panel h2 { font-size: 15px; margin: 0 0 12px; }
  canvas { max-height: 280px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid #f0f0f0; }
  th { background: #fafafa; font-weight: 600; color: #1d1d1f; text-align: right; position: sticky; top: 0; }
  th:first-child, td:first-child { text-align: left; }
  th:nth-child(2), td:nth-child(2) { text-align: left; }
  tr:hover td { background: #fafafa; }
  .scroll { max-height: 480px; overflow-y: auto; border-radius: 6px; border: 1px solid #f0f0f0; }
  .tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; }
  .tag-sl { background: #fde7e7; color: #d8302a; }
  .tag-tp { background: #e3f4e8; color: #2db849; }
  .tag-eod { background: #eaeaef; color: #6e6e73; }
  .tag-last { background: #eaeaef; color: #6e6e73; }
  .tag-yes { background: #e0eaff; color: #0050d8; }
  .tag-no { background: #f0f0f0; color: #999; }
  @media (max-width: 800px) { .row { grid-template-columns: 1fr; } }
  /* Two-column layout: main + right history sidebar */
  .layout { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
  @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } }
  .gate-banner { padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 12px;
                 display: flex; align-items: center; gap: 12px; }
  .gate-banner.pass { background: #e8f5e9; color: #1b5e20; border: 1px solid #81c784; }
  .gate-banner.block { background: #ffebee; color: #b71c1c; border: 1px solid #ef5350; }
  .gate-banner.unknown { background: #f5f5f5; color: #616161; border: 1px solid #bdbdbd; }
  .gate-banner .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .gate-banner.pass .dot { background: #2e7d32; }
  .gate-banner.block .dot { background: #c62828; }
  .gate-banner.unknown .dot { background: #9e9e9e; }
  .gate-banner .title { font-weight: 600; }
  .gate-banner .metrics { font-family: ui-monospace, monospace; font-size: 12px; opacity: 0.9; }
  .today-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
  @media (max-width: 900px) { .today-grid { grid-template-columns: 1fr; } }
  .today-card { padding: 10px 12px; background: #fafafa; border-radius: 6px; }
  .today-card .head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 6px; }
  .today-card .name { font-weight: 600; font-size: 13px; }
  .today-card .pnl { font-size: 16px; font-weight: 600; }
  .today-card .meta { font-size: 11px; color: #86868b; margin-bottom: 4px; }
  .today-card .pos { font-size: 11px; padding: 4px 0; border-top: 1px solid #eee; }
  .watch-grid { display:flex; gap:8px; flex-wrap:wrap; font-size:12px }
  .watch-pill { padding: 4px 10px; border-radius: 14px; background: #f0f0f0; display:inline-flex; gap:6px; align-items:center }
  .watch-pill.blocked { background: #fde7e7; color: #d8302a; }
  .watch-pill.attn { background: #fff3d6; color: #b85e00; }
  .history-list { font-size: 12px; max-height: 80vh; overflow-y: auto; }
  .history-item { padding: 6px 8px; border-bottom: 1px solid #f0f0f0; display:grid; grid-template-columns: auto auto 1fr auto; gap: 6px; align-items: baseline }
  .history-item .h-day { color: #86868b; font-size: 11px; }
  .history-item .h-sym { font-weight: 600; }
  .history-item .h-pnl { font-weight: 600; }
  .history-item.day-header { background: #fafafa; font-weight: 600; grid-template-columns: 1fr; }
  .strat-badge { display:inline-block; padding:1px 5px; border-radius:4px; font-size:10px; font-weight:600; text-align:center; min-width:22px; }
  .strat-global6  { background:#e0eaff; color:#0050d8; }
  .strat-per_group { background:#fff0d9; color:#c46a00; }
  .strat-per_stock { background:#e8e2ff; color:#6a3fd8; }
</style>
</head>
<body>

<h1>Auto-Trade Dashboard</h1>
<div class="subtitle">短only / SL=2.5% / TP=3% / 手續費 0.2355% / 80 萬日預算 / 1 張/筆 / -10k 熔斷</div>

<div class="filter" style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
  <div>月份: <select id="monthSel"><option value="all">全部</option></select></div>
  <div id="universeBox" style="background:white;padding:10px 14px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
    <div style="font-size:11px;color:#86868b;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px">下單清單 (Active)</div>
    <div id="universeChecks" style="display:flex;gap:10px;flex-wrap:wrap;font-size:13px"></div>
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
      <button id="saveUniverse" style="font-size:12px;padding:4px 10px;border-radius:4px;border:1px solid #0050d8;background:#0050d8;color:white;cursor:pointer">儲存</button>
      <input id="addSymbol" type="text" placeholder="加新標的 (例 2330)" maxlength="6"
             style="font-size:12px;padding:4px 8px;border-radius:4px;border:1px solid #d2d2d7;width:120px">
      <button id="addBtn" style="font-size:12px;padding:4px 10px;border-radius:4px;border:1px solid #d2d2d7;background:white;cursor:pointer">加入候選</button>
      <span id="universeStatus" style="font-size:11px;color:#86868b"></span>
    </div>
  </div>
</div>

<div class="layout">
<div class="main-col">

<div class="panel" style="margin-bottom:16px">
  <h2>今日觀察清單 <span id="todayDay" style="font-size:11px;color:#86868b;font-weight:normal"></span></h2>
  <div id="taiexGate" class="gate-banner unknown" title="TAIEX regime gate — 強勢盤+跳空時自動阻擋當日 short">
    <span class="dot"></span>
    <span class="title">TAIEX Regime Gate</span>
    <span id="taiexGateState">載入中…</span>
    <span class="metrics" id="taiexGateMetrics"></span>
  </div>
  <div id="watchlist" class="watch-grid"></div>
</div>

<div class="panel" style="margin-bottom:16px">
  <h2>當日進場(3 策略並行)</h2>
  <div id="todayGrid" class="today-grid"></div>
</div>

<div class="cards" id="cards"></div>

<div class="row">
  <div class="panel">
    <h2>月損益</h2>
    <canvas id="monthChart"></canvas>
  </div>
  <div class="panel">
    <h2>累計權益曲線</h2>
    <canvas id="equityChart"></canvas>
  </div>
</div>

<div class="panel" style="margin-bottom: 16px;">
  <h2>交易紀錄 (Filled Trades)</h2>
  <div class="scroll">
    <table id="tradesTbl">
      <thead><tr>
        <th>#</th><th>時間</th><th>標的</th><th>進場</th><th>出場</th><th>出場原因</th>
        <th>信心度</th><th>NTD 損益</th><th>fold</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="panel" style="margin-bottom: 16px;">
  <h2>Paper Trading (即時/盤後)</h2>
  <div style="margin-bottom:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <div>日期: <select id="paperDay"><option value="">-- 選日期 --</option></select></div>
    <div>策略: <select id="paperStrategy">
      <option value="global6">global6</option>
      <option value="per_group">per_group</option>
      <option value="per_stock">per_stock</option>
    </select></div>
    <div id="paperDayPnl" style="font-size:13px"></div>
  </div>
  <canvas id="paperDailyChart" style="max-height: 200px; margin-bottom: 12px"></canvas>
  <div class="scroll">
    <table id="paperTbl">
      <thead><tr>
        <th>#</th><th title="模型訊號產生時間">訊號時間</th><th>標的</th><th>狀態</th><th>信心度</th>
        <th title="實際模擬進場時間">進場時間</th><th>進場價</th>
        <th title="實際模擬出場時間">出場時間</th><th>出場價</th>
        <th title="hover 看詳細">出場原因</th><th title="進場至出場分鐘數">持倉(分)</th>
        <th>NTD 損益</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="panel">
  <h2>訊號源 (Raw Model Predictions) <span style="font-size:11px;font-weight:normal;color:#86868b">— 最新 1000 筆</span></h2>
  <div style="margin-bottom:8px">
    <label><input type="checkbox" id="trigOnly"> 只看觸發訊號</label>
  </div>
  <div class="scroll">
    <table id="sigTbl">
      <thead><tr>
        <th>#</th><th>時間</th><th>標的</th><th>收盤</th>
        <th>proba</th><th>信心度</th><th>實際 fwd_ret</th><th>觸發</th><th>fold</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

</div><!-- /main-col -->

<aside class="panel" style="position:sticky;top:16px;align-self:start;height:fit-content">
  <h2>歷史進場 <span style="font-size:11px;color:#86868b;font-weight:normal">— 最近 500 筆</span></h2>
  <div style="margin-bottom:8px;font-size:12px">
    過濾策略: <select id="historyStrat" style="font-size:12px;padding:2px 6px">
      <option value="">全部</option>
      <option value="global6">global6</option>
      <option value="per_group">per_group</option>
      <option value="per_stock">per_stock</option>
    </select>
    <span style="margin-left:8px">
      <span class="strat-badge strat-global6">G6</span>=global6
      <span class="strat-badge strat-per_group">PG</span>=per_group
      <span class="strat-badge strat-per_stock">PS</span>=per_stock
    </span>
  </div>
  <div id="historyList" class="history-list"></div>
</aside>

</div><!-- /layout -->

<script>
let monthChart, equityChart, paperDailyChart;
const fmt = n => (n >= 0 ? '+' : '') + n.toLocaleString();
const STRAT_LABEL = { global6: 'G6', per_group: 'PG', per_stock: 'PS' };

// Helper: append a row of cells (auto-escaped via textContent)
// Each cell can be a primitive or an object: {tag:'span', cls, text, title}
function addRow(tbody, cells) {
  const tr = document.createElement('tr');
  for (const c of cells) {
    const td = document.createElement('td');
    if (c && typeof c === 'object' && c.tag === 'span') {
      const span = document.createElement('span');
      span.className = c.cls || '';
      span.textContent = c.text;
      if (c.title) span.title = c.title;
      td.appendChild(span);
    } else if (c && typeof c === 'object') {
      td.textContent = c.text == null ? '' : String(c.text);
      if (c.cls) td.className = c.cls;
      if (c.title) td.title = c.title;
    } else {
      td.textContent = c == null ? '' : String(c);
    }
    tr.appendChild(td);
  }
  tbody.appendChild(tr);
}

function clearTbody(sel) {
  const tbody = document.querySelector(sel);
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  return tbody;
}

function setCard(label, value, cls) {
  const card = document.createElement('div');
  card.className = 'card';
  const l = document.createElement('div'); l.className = 'label'; l.textContent = label;
  const v = document.createElement('div'); v.className = 'value' + (cls ? ' ' + cls : '');
  v.textContent = value;
  card.appendChild(l); card.appendChild(v);
  return card;
}

async function reload() {
  const m = document.getElementById('monthSel').value;
  await Promise.all([loadCards(m), loadEquity(m), loadTrades(m), loadSignals()]);
}

async function loadCards(m) {
  const r = await fetch('/api/summary?month=' + encodeURIComponent(m)).then(r => r.json());
  const cards = document.getElementById('cards');
  while (cards.firstChild) cards.removeChild(cards.firstChild);
  if (r.empty) { cards.appendChild(setCard('資料', '無')); return; }
  const cls = v => v >= 0 ? 'pos' : 'neg';
  cards.appendChild(setCard('累計損益', fmt(r.total_pnl), cls(r.total_pnl)));
  cards.appendChild(setCard('勝率', r.win_rate + '%'));
  cards.appendChild(setCard('交易數', r.trades));
  cards.appendChild(setCard('平均/筆', fmt(r.avg_pnl), cls(r.avg_pnl)));
  cards.appendChild(setCard('最大回撤', fmt(r.mdd), 'neg'));
  cards.appendChild(setCard('PF', r.pf));
  cards.appendChild(setCard('SL / TP / EOD', r.sl_count + ' / ' + r.tp_count + ' / ' + r.eod_count));
  cards.appendChild(setCard('最大獲利 / 虧損', fmt(r.max_win) + ' / ' + fmt(r.max_loss)));
}

async function loadMonthly() {
  const r = await fetch('/api/monthly').then(r => r.json());
  const sel = document.getElementById('monthSel');
  r.forEach(x => { const o = document.createElement('option'); o.value = x.month; o.textContent = x.month; sel.appendChild(o); });
  if (monthChart) monthChart.destroy();
  monthChart = new Chart(document.getElementById('monthChart'), {
    type: 'bar',
    data: {
      labels: r.map(x => x.month),
      datasets: [{
        label: 'NTD 損益',
        data: r.map(x => x.ntd_pnl),
        backgroundColor: r.map(x => x.ntd_pnl >= 0 ? '#2db849' : '#d8302a'),
      }]
    },
    options: { plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => `${fmt(c.parsed.y)} (${r[c.dataIndex].trades} 筆, 勝率 ${r[c.dataIndex].win_rate}%)` } }
    } }
  });
}

async function loadEquity(m) {
  const r = await fetch('/api/equity?month=' + encodeURIComponent(m)).then(r => r.json());
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(document.getElementById('equityChart'), {
    type: 'line',
    data: { labels: r.map(x => x.signal_dt.slice(0,16)),
            datasets: [{ label: '累計 NTD', data: r.map(x => x.cum_pnl), borderColor: '#0050d8', fill: false, pointRadius: 1, tension: 0.1 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 8 } } } }
  });
}

async function loadTrades(m) {
  const r = await fetch('/api/trades?month=' + encodeURIComponent(m)).then(r => r.json());
  const tbody = clearTbody('#tradesTbl tbody');
  r.forEach((t, i) => {
    addRow(tbody, [
      i+1, t.signal_dt.slice(0,16), t.symbol, t.entry, t.exit,
      { tag:'span', cls:'tag tag-' + t.reason, text: t.reason },
      t.conf,
      { tag:'span', cls: t.ntd_pnl >= 0 ? 'pos' : 'neg', text: fmt(t.ntd_pnl) },
      t.fold,
    ]);
  });
}

async function loadSignals() {
  const m = document.getElementById('monthSel').value;
  const trig = document.getElementById('trigOnly').checked ? '1' : '0';
  const r = await fetch('/api/signals?month=' + encodeURIComponent(m) + '&triggered=' + trig).then(r => r.json());
  const tbody = clearTbody('#sigTbl tbody');
  r.forEach((s, i) => {
    addRow(tbody, [
      i+1, s.datetime.slice(0,16), s.symbol, s.close,
      s.proba.toFixed(3), s.conf.toFixed(3),
      { tag:'span', cls: s.fwd_ret >= 0 ? 'pos' : 'neg', text: (s.fwd_ret*100).toFixed(2) + '%' },
      { tag:'span', cls: s.triggered ? 'tag tag-yes' : 'tag tag-no', text: s.triggered ? 'YES' : 'no' },
      s.fold,
    ]);
  });
}

document.getElementById('monthSel').addEventListener('change', reload);
document.getElementById('trigOnly').addEventListener('change', loadSignals);

async function loadPaperSummary() {
  // Days come from anywhere there are CSV files (live trading included)
  const allDays = await fetch('/api/paper_strategies').then(r => r.json()).catch(_ => ({days:[]}));
  const eod = await fetch('/api/paper_summary').then(r => r.json()).catch(_ => []);
  const eodByDay = Object.fromEntries(eod.map(x => [x.date, x]));
  const sel = document.getElementById('paperDay');
  while (sel.options.length > 1) sel.remove(1);
  (allDays.days || []).forEach(day => {
    const o = document.createElement('option'); o.value = day;
    const eodRow = eodByDay[day];
    const tag = eodRow ? '  (EOD ' + (eodRow.ntd_pnl >= 0 ? '+' : '') + Number(eodRow.ntd_pnl).toLocaleString() + ')' : '';
    o.textContent = day + tag;
    sel.appendChild(o);
  });
  const r = eod;
  if (paperDailyChart) paperDailyChart.destroy();
  paperDailyChart = new Chart(document.getElementById('paperDailyChart'), {
    type: 'bar',
    data: {
      labels: r.map(x => x.date),
      datasets: [{
        label: 'Paper Daily PnL', data: r.map(x => Number(x.ntd_pnl)),
        backgroundColor: r.map(x => x.ntd_pnl >= 0 ? '#2db849' : '#d8302a'),
      }]
    },
    options: { plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => `${fmt(c.parsed.y)} (filled ${r[c.dataIndex].filled}, win ${(r[c.dataIndex].win_rate*100).toFixed(0)}%)` } }
    } }
  });
  // auto-select latest
  if (r.length) {
    sel.value = r[r.length - 1].date;
    await loadPaperTrades();
  }
}

async function loadPaperTrades() {
  const day = document.getElementById('paperDay').value;
  const strat = document.getElementById('paperStrategy').value;
  const tbody = clearTbody('#paperTbl tbody');
  const pnlBox = document.getElementById('paperDayPnl');
  if (pnlBox) pnlBox.textContent = '';
  if (!day) return;
  const r = await fetch('/api/paper_trades?day=' + encodeURIComponent(day) + '&strategy=' + encodeURIComponent(strat)).then(r => r.json());
  if (pnlBox) {
    const filled = r.filter(t => t.status === 'filled' || t.status === 'closed');
    const pnl = filled.reduce((s,t) => s + Number(t.ntd_pnl || 0), 0);
    const wins = filled.filter(t => Number(t.ntd_pnl) > 0).length;
    if (filled.length) {
      pnlBox.innerHTML = '';
      const cls = pnl >= 0 ? 'pos' : 'neg';
      const span = document.createElement('span');
      span.className = cls;
      span.textContent = (pnl >= 0 ? '+' : '') + pnl.toLocaleString();
      pnlBox.appendChild(document.createTextNode(filled.length + ' 筆,勝率 ' + (wins/filled.length*100).toFixed(0) + '%,NTD '));
      pnlBox.appendChild(span);
    } else {
      pnlBox.textContent = '當日無成交';
    }
  }
  r.forEach((t, i) => {
    const reason = t.exit_reason || '';
    const statusOk = t.status === 'filled' || t.status === 'closed';
    const fmtDt = s => !s ? '' : String(s).slice(11, 16) || String(s).slice(0, 16);
    addRow(tbody, [
      i+1,
      { text: (t.datetime || '').slice(0, 16), title: t.entry_reason || '' },
      t.symbol,
      { tag:'span', cls: statusOk ? 'tag tag-yes' : 'tag tag-no',
        text: t.status, title: t.entry_reason || '' },
      Number(t.conf).toFixed(3),
      fmtDt(t.entry_dt || ''),
      t.entry_price || '',
      fmtDt(t.exit_dt || ''),
      t.exit_price || '',
      reason ? { tag:'span', cls:'tag tag-' + reason, text: reason, title: t.exit_detail || '' } : '',
      t.holding_min !== '' && t.holding_min !== undefined && t.holding_min !== null ? t.holding_min : '',
      t.ntd_pnl !== '' && t.ntd_pnl !== undefined
        ? { tag:'span', cls: Number(t.ntd_pnl) >= 0 ? 'pos' : 'neg', text: fmt(Number(t.ntd_pnl)) }
        : '',
    ]);
  });
}

document.getElementById('paperDay').addEventListener('change', loadPaperTrades);
document.getElementById('paperStrategy').addEventListener('change', loadPaperTrades);

async function loadUniverse() {
  const r = await fetch('/api/universe').then(r => r.json());
  const active = new Set(r.active_symbols || []);
  const candidates = r.candidates || [];
  const box = document.getElementById('universeChecks');
  while (box.firstChild) box.removeChild(box.firstChild);
  candidates.forEach(sym => {
    const lbl = document.createElement('label');
    lbl.style.cursor = 'pointer';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = sym; cb.checked = active.has(sym);
    cb.dataset.symbol = sym;
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + sym));
    box.appendChild(lbl);
  });
}

async function saveUniverse() {
  const checks = document.querySelectorAll('#universeChecks input[type=checkbox]:checked');
  const active = Array.from(checks).map(c => c.value);
  const status = document.getElementById('universeStatus');
  status.textContent = '儲存中...';
  const r = await fetch('/api/universe', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({active_symbols: active})
  }).then(r => r.json());
  status.textContent = '已儲存 (' + r.active_symbols.length + ' 檔)';
  setTimeout(() => status.textContent = '', 3000);
}

async function addCandidate() {
  const inp = document.getElementById('addSymbol');
  const sym = inp.value.trim();
  if (!/^\d{4,6}$/.test(sym)) { alert('股票代碼格式錯誤'); return; }
  const r = await fetch('/api/universe').then(r => r.json());
  const cands = new Set(r.candidates || []);
  if (cands.has(sym)) { alert(sym + ' 已在候選中'); return; }
  cands.add(sym);
  // POST a new payload that also extends candidates — backend currently rejects unknowns,
  // so we'll write candidates via a dedicated endpoint or by patching the JSON directly.
  // Quick path: re-fetch + write via /api/universe_candidates
  const resp = await fetch('/api/universe_candidates', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({symbol: sym})
  }).then(r => r.json());
  if (resp.error) { alert(resp.error); return; }
  inp.value = '';
  await loadUniverse();
}

document.getElementById('saveUniverse').addEventListener('click', saveUniverse);
document.getElementById('addBtn').addEventListener('click', addCandidate);
document.getElementById('addSymbol').addEventListener('keydown', e => { if (e.key === 'Enter') addCandidate(); });

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v * 100 >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
}

function renderTaiexGate(g) {
  const banner = document.getElementById('taiexGate');
  const stateEl = document.getElementById('taiexGateState');
  const metricsEl = document.getElementById('taiexGateMetrics');
  banner.classList.remove('pass', 'block', 'unknown');
  if (!g || g.reason === 'unavailable' || g.reason.startsWith('error') ||
      g.reason.startsWith('no_taiex') || g.reason.startsWith('empty_intraday') ||
      g.reason.startsWith('insufficient')) {
    banner.classList.add('unknown');
    stateEl.textContent = g && g.reason ? '尚未就緒(' + g.reason + ')' : '尚未就緒';
    metricsEl.textContent = '';
    return;
  }
  if (g.block) {
    banner.classList.add('block');
    stateEl.textContent = '🛑 阻擋當日 short';
  } else {
    banner.classList.add('pass');
    stateEl.textContent = '✓ 通行';
  }
  metricsEl.textContent = ' · 5d trend=' + fmtPct(g.trend) + '  open gap=' + fmtPct(g.gap) +
                          '  (門檻 trend>+2.00% AND gap>+0.50%)';
}

async function loadToday() {
  const r = await fetch('/api/today').then(r => r.json());
  document.getElementById('todayDay').textContent = '— ' + r.day;
  renderTaiexGate(r.taiex_gate);

  // Watchlist
  const watch = document.getElementById('watchlist');
  while (watch.firstChild) watch.removeChild(watch.firstChild);
  if (!r.watchlist || !r.watchlist.length) {
    const empty = document.createElement('div'); empty.textContent = '尚未載入(請先跑 fetch_blacklist)';
    empty.style.color = '#86868b'; empty.style.fontSize = '12px';
    watch.appendChild(empty);
  } else {
    r.watchlist.forEach(s => {
      const pill = document.createElement('div');
      let cls = 'watch-pill';
      if (s.blocked) cls += ' blocked';
      else if (s.is_attention) cls += ' attn';
      pill.className = cls;
      const txt = s.symbol + (s.name ? ' ' + s.name : '') +
                  (s.ref_price ? ' @' + s.ref_price : '') +
                  (s.blocked ? ' (BLOCKED)' : (s.is_attention ? ' (注意)' : ''));
      pill.textContent = txt;
      watch.appendChild(pill);
    });
  }

  // Today entries per strategy
  const grid = document.getElementById('todayGrid');
  while (grid.firstChild) grid.removeChild(grid.firstChild);
  ['global6', 'per_group', 'per_stock'].forEach(name => {
    const s = (r.strategies || {})[name] || {open:[], closed:[], daily_pnl:0, budget_used:0, breaker_tripped:false};
    const card = document.createElement('div'); card.className = 'today-card';
    const head = document.createElement('div'); head.className = 'head';
    const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = name;
    const pnl = document.createElement('span'); pnl.className = 'pnl ' + (s.daily_pnl >= 0 ? 'pos' : 'neg');
    pnl.textContent = fmt(s.daily_pnl);
    head.appendChild(nm); head.appendChild(pnl);
    card.appendChild(head);

    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = `成交 ${s.closed.length}  持倉 ${s.open.length}  預算 ${Math.round(s.budget_used).toLocaleString()}` +
                       (s.breaker_tripped ? '  🛑 熔斷' : '');
    card.appendChild(meta);

    // Open positions first (most relevant)
    s.open.forEach(p => {
      const row = document.createElement('div'); row.className = 'pos';
      row.title = p.entry_reason || '';
      row.textContent = `🟢 ${p.symbol} @${p.entry_price} sl=${(p.sl_px||0).toFixed(2)} tp=${(p.tp_px||0).toFixed(2)}`;
      card.appendChild(row);
    });
    // Closed today
    s.closed.slice(-8).reverse().forEach(p => {
      const row = document.createElement('div'); row.className = 'pos';
      const pcls = (p.ntd_pnl || 0) >= 0 ? 'pos' : 'neg';
      row.title = p.exit_detail || '';
      const tag = document.createElement('span');
      tag.className = 'tag tag-' + (p.exit_reason || 'last');
      tag.textContent = p.exit_reason || '?';
      const txt = document.createTextNode(` ${p.symbol} ${p.entry_price}→${p.exit_price} `);
      const v = document.createElement('span'); v.className = pcls; v.textContent = fmt(Math.round(p.ntd_pnl||0));
      row.appendChild(tag); row.appendChild(txt); row.appendChild(v);
      card.appendChild(row);
    });
    if (!s.open.length && !s.closed.length) {
      const empty = document.createElement('div'); empty.className = 'pos'; empty.style.color = '#86868b';
      empty.textContent = '今日尚無訊號';
      card.appendChild(empty);
    }
    grid.appendChild(card);
  });
}

async function loadHistory() {
  const filter = document.getElementById('historyStrat').value;
  const data = await fetch('/api/history_entries').then(r => r.json());
  const filtered = filter ? data.filter(t => t.strategy === filter) : data;
  const box = document.getElementById('historyList');
  while (box.firstChild) box.removeChild(box.firstChild);
  let lastDay = null;
  filtered.forEach(t => {
    if (t.day !== lastDay) {
      const h = document.createElement('div'); h.className = 'history-item day-header';
      h.textContent = t.day;
      box.appendChild(h);
      lastDay = t.day;
    }
    const row = document.createElement('div');
    row.className = 'history-item';
    row.title = `${t.strategy} | entry ${t.entry_dt} → exit ${t.exit_dt} (${t.holding_min}m) | ${t.entry_price} → ${t.exit_price} | ${t.exit_reason}`;
    const c1 = document.createElement('span'); c1.className = 'h-day'; c1.textContent = (t.entry_dt || '');
    const sb = document.createElement('span'); sb.className = 'strat-badge strat-' + t.strategy;
    sb.textContent = STRAT_LABEL[t.strategy] || t.strategy;
    const c2 = document.createElement('span'); c2.className = 'h-sym'; c2.textContent = t.symbol;
    const c3 = document.createElement('span'); c3.className = 'h-pnl ' + (t.ntd_pnl >= 0 ? 'pos' : 'neg');
    c3.textContent = fmt(t.ntd_pnl);
    row.appendChild(c1); row.appendChild(sb); row.appendChild(c2); row.appendChild(c3);
    box.appendChild(row);
  });
  if (!filtered.length) {
    const empty = document.createElement('div'); empty.textContent = '尚無歷史紀錄'; empty.style.color = '#86868b';
    box.appendChild(empty);
  }
}

document.getElementById('historyStrat').addEventListener('change', loadHistory);

(async () => {
  await loadMonthly();
  await loadPaperSummary();
  await loadUniverse();
  await loadToday();
  await loadHistory();
  await reload();
  // Auto-refresh today panel every 30s during market hours
  setInterval(() => { loadToday(); loadHistory(); }, 30000);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 0.0.0.0 so the bind works from outside the docker container too.
    # Safe because: (a) compose only maps to host port 5050, (b) read-only model
    # outputs; only universe.json mutates and that's intentional.
    app.run(host="0.0.0.0", port=5050, debug=False)
