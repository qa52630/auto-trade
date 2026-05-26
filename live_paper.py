"""Intraday paper trader — runs during market hours (09:00-13:30) and simulates
fills in real time using Fugle intraday 1-min candles.

Each cycle (every 60s):
  1. Poll candles for each symbol in data/universe.json -> data/intraday_1m/{day}/{sym}.csv
  2. Rebuild 5-min features (today + historical context)
  3. Apply the same filter chain as paper_trade.py
  4. For new short signals -> open virtual position at next 1-min open
  5. Walk open positions against the latest tick: SL=2.5% / TP=3% / 13:25 force-close
  6. Atomically rewrite data/paper_trades/{day}.csv + state_{day}.json
  7. Sleep to next minute

State persistence ensures a crash mid-day doesn't lose open positions.

Run:    python3 live_paper.py
Stop:   ctrl-C (positions will be force-closed at 13:25 by the loop anyway)
"""
from __future__ import annotations

import itertools
import json
import os
import signal as sigmod
import sys
import threading
import time
from datetime import datetime, time as dtime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train as T

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
PER_GROUP_DIR = ROOT / "artifacts" / "models_per_group"
PER_STOCK_DIR = ROOT / "artifacts" / "models_per_stock"
STOCKS_JSON = ROOT / "stocks.json"
UNIVERSE_JSON = ROOT / "data" / "universe.json"
INTRADAY_DIR = ROOT / "data" / "intraday_1m"
PAPER_DIR = ROOT / "data" / "paper_trades"
BLACKLIST_DIR = ROOT / "data" / "blacklist"

TZ = timezone(timedelta(hours=8))
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(13, 30)
EOD_FORCE_CLOSE = dtime(13, 25)
POLL_SEC = 60

# Strategy params (mirror paper_trade.py)
SHORT_SL = 0.025
SHORT_TP = 0.030
COST_RATE = 0.002355
PROBA_SHORT_THRESHOLD = 0.35
SHARES_PER_LOT = 1000
DAILY_BUDGET = 800_000
DAILY_LOSS_CAP = 10_000
MAX_INTRADAY_PCT_FOR_SHORT = 0.07
HOLD_BARS = 6
THRESHOLD = 0.005


def _now() -> datetime:
    return datetime.now(TZ)


def _today() -> str:
    return _now().date().isoformat()


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


LOGIN_RETRY = 3
LOGIN_BACKOFF = 30  # seconds


def _build_sdk_once():
    _load_env()
    from taishin_sdk import TaishinSDK
    api_url = os.environ.get("TAISHIN_API_URL", "https://fugletrade.tssco.com.tw")
    sdk = TaishinSDK(api_url=api_url)
    pfx = next(ROOT.glob("*.pfx"))
    accounts = sdk.login(
        os.environ["API_USER"], os.environ["API_PASS"], str(pfx), os.environ["API_CERT_PASS"]
    )
    if not accounts:
        raise RuntimeError("login returned no accounts")
    sdk.init_realtime(accounts[0])
    return sdk


def _build_sdk(retries: int = LOGIN_RETRY):
    """Login with retries; raise SystemExit only after all retries fail."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            sdk = _build_sdk_once()
            if attempt > 1:
                print(f"[auth] login OK on attempt {attempt}", file=sys.stderr)
            return sdk
        except Exception as e:
            last_err = e
            print(f"[auth] attempt {attempt}/{retries} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            if attempt < retries:
                time.sleep(LOGIN_BACKOFF)
    raise SystemExit(f"login failed after {retries} attempts: {last_err}")


def load_universe() -> list[str]:
    if not UNIVERSE_JSON.exists():
        return ["6770", "2337", "1802", "2408", "3481", "1815"]
    return json.loads(UNIVERSE_JSON.read_text())["active_symbols"]


def load_blacklist(day: str) -> set[str]:
    p = BLACKLIST_DIR / f"{day}.json"
    if not p.exists():
        return set()
    try:
        return set(str(c) for c in json.loads(p.read_text()).get("no_short_sell", []))
    except json.JSONDecodeError:
        return set()


def fetch_candles_for(sdk, symbol: str) -> pd.DataFrame:
    """Return today's 1-min candles as a DataFrame with cols dt,open,high,low,close,volume."""
    r = sdk.marketdata.rest_client.stock.intraday.candles(symbol=symbol, timeframe="1")
    bars = r.get("data") or []
    if not bars:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df["dt"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["dt", "open", "high", "low", "close", "volume"]].sort_values("dt").reset_index(drop=True)


def save_intraday(day: str, symbol: str, df: pd.DataFrame) -> Path:
    d = INTRADAY_DIR / day
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{symbol}.csv"
    df.to_csv(p, index=False)
    return p


TAIEX_DAILY_CSV = ROOT / "data" / "taiex" / "daily.csv"
TAIEX_INTRADAY_DIR = ROOT / "data" / "taiex" / "intraday_1m"
# Regime gate thresholds — only block short entries when BOTH conditions hold.
# Calibrated on 124d TAIEX history: triggers ~7% of days, 8/9 closed up.
TAIEX_GATE_TREND_THR = 0.02   # 5-day trend (close_{D-1} vs close_{D-6})
TAIEX_GATE_GAP_THR = 0.005    # today's open vs prev close


def taiex_regime_gate(day: str) -> tuple[bool, dict]:
    """Return (block_shorts, info). Reads daily.csv for D-1 history and today's
    intraday_1m first bar for today's open. Both must be present for the gate
    to fire — if data is missing, default to NOT blocking (fail-open).
    """
    info = {"trend": None, "gap": None, "reason": ""}
    if not TAIEX_DAILY_CSV.exists():
        info["reason"] = "no_taiex_daily"; return False, info
    daily = pd.read_csv(TAIEX_DAILY_CSV).sort_values("date").reset_index(drop=True)
    daily["date"] = daily["date"].astype(str)
    # Trend uses close_{D-1} and close_{D-6}; both must exist BEFORE today.
    prior = daily[daily["date"] < day].tail(6)
    if len(prior) < 6:
        info["reason"] = "insufficient_taiex_history"; return False, info
    prev_close = float(prior.iloc[-1]["close"])
    close_d6 = float(prior.iloc[0]["close"])
    trend = (prev_close - close_d6) / close_d6
    info["trend"] = round(trend, 5)
    # Gap needs today's TAIEX open from intraday_1m first bar.
    ip = TAIEX_INTRADAY_DIR / f"{day}.csv"
    if not ip.exists():
        info["reason"] = "no_taiex_intraday"; return False, info
    intra = pd.read_csv(ip)
    if intra.empty:
        info["reason"] = "empty_intraday"; return False, info
    today_open = float(intra.iloc[0]["open"])
    gap = (today_open - prev_close) / prev_close
    info["gap"] = round(gap, 5)
    block = (trend > TAIEX_GATE_TREND_THR) and (gap > TAIEX_GATE_GAP_THR)
    info["reason"] = (
        f"block: trend={trend*100:+.2f}% > {TAIEX_GATE_TREND_THR*100:.1f}% AND "
        f"gap={gap*100:+.2f}% > {TAIEX_GATE_GAP_THR*100:.1f}%"
        if block else
        f"pass: trend={trend*100:+.2f}% gap={gap*100:+.2f}%"
    )
    return block, info


def build_features_for_today(day: str, symbols: list[str]) -> pd.DataFrame:
    """Combine today's intraday cache with historical 1-min CSVs and run feature pipeline."""
    df1_hist = T.load_all()
    # Append today's bars; match historical tz dtype (datetime64[us, Asia/Taipei])
    hist_tz = df1_hist["datetime"].dt.tz if hasattr(df1_hist["datetime"], "dt") and df1_hist["datetime"].dt.tz else "Asia/Taipei"
    today_dfs = []
    for sym in symbols:
        p = INTRADAY_DIR / day / f"{sym}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["datetime"] = pd.to_datetime(d["dt"]).dt.tz_localize(hist_tz)
        d["symbol"] = sym
        d["date"] = day
        today_dfs.append(d[["symbol", "datetime", "date", "open", "high", "low", "close", "volume"]])
    if today_dfs:
        td = pd.concat(today_dfs, ignore_index=True)
        # Avoid duplicating dates already in historical
        existing_dates = set(df1_hist["date"].unique())
        if day in existing_dates:
            df1_hist = df1_hist[df1_hist["date"] != day]
        df1 = pd.concat([df1_hist, td], ignore_index=True)
    else:
        df1 = df1_hist

    df5 = T.resample_5m(df1)
    feats = T.build_features(df5)
    feats = T.add_cross_sectional(feats, ROOT / "stocks.json")
    feats = feats[feats["symbol"].isin(symbols)]
    feats = T.label_next(feats, threshold=THRESHOLD, hold_bars=HOLD_BARS)
    feats = feats.dropna(subset=T.FEATURE_COLS)
    feats = feats[feats["date"] == day]
    # Liquidity gate (entry bar only — exit bar not yet observed)
    feats = feats[(feats["volume"] >= 1000) & (feats["high"] != feats["low"])]
    return feats


def predict(model, feats: pd.DataFrame) -> pd.DataFrame:
    if feats.empty:
        feats = feats.copy()
        feats["proba"] = []; feats["conf"] = []
        return feats
    proba = model.predict_proba(feats[T.FEATURE_COLS].values)[:, 1]
    feats = feats.copy()
    feats["proba"] = proba
    feats["conf"] = 1 - proba
    feats["triggered"] = proba <= PROBA_SHORT_THRESHOLD
    return feats


def _state_path(day: str, strategy: str = "global6") -> Path:
    suffix = "" if strategy == "global6" else f"_{strategy}"
    return PAPER_DIR / f"state_{day}{suffix}.json"


def _trades_path(day: str, strategy: str = "global6") -> Path:
    suffix = "" if strategy == "global6" else f"_{strategy}"
    return PAPER_DIR / f"{day}{suffix}.csv"


def load_state(day: str, strategy: str = "global6") -> dict:
    p = _state_path(day, strategy)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "open_positions": [],
        "closed_trades": [],
        "skipped": [],
        "budget_used": 0.0,
        "daily_pnl": 0.0,
        "breaker_tripped": False,
        "processed_signals": [],
    }


_tmp_counter = itertools.count()


def _unique_tmp(p: Path) -> Path:
    """Per-writer unique tmp path (pid+tid+counter) so concurrent writers /
    macOS bind-mount FS lag never share or vanish a tmp file."""
    return p.parent / f"{p.name}.{os.getpid()}.{threading.get_ident()}.{next(_tmp_counter)}.tmp"


def _atomic_replace(tmp: Path, dst: Path, retries: int = 5) -> None:
    """tmp.replace(dst) with retry — VirtioFS/gRPC-FUSE occasionally raises
    ENOENT on a just-written file. If tmp already consumed and dst exists,
    treat as success."""
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            tmp.replace(dst)
            return
        except FileNotFoundError as e:
            last_err = e
            if not tmp.exists() and dst.exists():
                return
            time.sleep(0.05)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    if last_err is not None:
        raise last_err


def save_state(day: str, state: dict, strategy: str = "global6") -> None:
    p = _state_path(day, strategy)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(p)
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    _atomic_replace(tmp, p)


def write_paper_csv(day: str, state: dict, strategy: str = "global6") -> None:
    """Rebuild today's paper_trades CSV from state."""
    rows = list(state["closed_trades"])
    # Mark open positions as 'open' for visibility
    for op in state["open_positions"]:
        rows.append({**op, "status": "open"})
    for s in state["skipped"]:
        rows.append(s)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("datetime", na_position="last").reset_index(drop=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    p = _trades_path(day, strategy)
    tmp = _unique_tmp(p)
    df.to_csv(tmp, index=False)
    _atomic_replace(tmp, p)


def open_position(state: dict, sig: dict, entry_price: float, entry_dt: str) -> None:
    notional = entry_price * SHARES_PER_LOT
    sl_px = round(entry_price * (1 + SHORT_SL), 4)
    tp_px = round(entry_price * (1 - SHORT_TP), 4)
    state["open_positions"].append({
        "datetime": sig["datetime"], "symbol": sig["symbol"],
        "signal_close": sig["close"], "conf": sig["conf"], "proba": sig["proba"],
        "entry_dt": entry_dt, "entry_price": entry_price,
        "sl_px": sl_px, "tp_px": tp_px,
        "lots": 1, "notional": notional,
        "status": "open",
        "entry_reason": f"short proba={sig['proba']:.3f} ≤ {PROBA_SHORT_THRESHOLD} (conf={sig['conf']:.3f}); "
                        f"filters passed: not_breaker, not_blacklist, intraday_pct_ok, budget_ok",
        "signal_dt": sig["datetime"],
    })
    state["budget_used"] += notional
    state["processed_signals"].append([sig["symbol"], sig["datetime"]])


def close_position(state: dict, pos: dict, exit_price: float, exit_dt: str,
                   reason: str, exit_detail: str = "") -> None:
    gross_ret = (pos["entry_price"] - exit_price) / pos["entry_price"]
    net_ret = gross_ret - COST_RATE
    ntd_pnl = pos["notional"] * net_ret
    # Holding duration
    try:
        e_dt = pd.to_datetime(pos["entry_dt"]); x_dt = pd.to_datetime(exit_dt)
        if e_dt.tz is not None: e_dt = e_dt.tz_localize(None)
        if x_dt.tz is not None: x_dt = x_dt.tz_localize(None)
        holding_min = round((x_dt - e_dt).total_seconds() / 60.0, 1)
    except Exception:
        holding_min = None
    closed = {**pos,
              "status": "closed",
              "exit_dt": exit_dt, "exit_price": exit_price, "exit_reason": reason,
              "exit_detail": exit_detail or _default_exit_detail(reason, pos, exit_price),
              "holding_min": holding_min,
              "gross_ret": round(gross_ret, 5), "net_ret": round(net_ret, 5),
              "ntd_pnl": round(ntd_pnl, 2)}
    state["closed_trades"].append(closed)
    state["daily_pnl"] += ntd_pnl
    if state["daily_pnl"] <= -DAILY_LOSS_CAP:
        state["breaker_tripped"] = True
    print(f"  [CLOSE] {pos['symbol']} {pos['entry_price']:.2f}→{exit_price:.2f} {reason} "
          f"hold={holding_min}m NTD{ntd_pnl:+,.0f}  daily={state['daily_pnl']:+,.0f}",
          file=sys.stderr)


def _default_exit_detail(reason: str, pos: dict, exit_price: float) -> str:
    if reason == "sl":
        return f"stop-loss: price reached sl_px={pos['sl_px']:.4f} (entry × 1.025)"
    if reason == "tp":
        return f"take-profit: price reached tp_px={pos['tp_px']:.4f} (entry × 0.97)"
    if reason == "eod":
        return f"force close at 13:25 EOD (no SL/TP hit)"
    if reason == "last":
        return f"end of data; closed at last available bar close"
    return reason


def process_open_positions(state: dict, candle_cache: dict[str, pd.DataFrame], now_dt: datetime) -> None:
    """Walk each open position against latest bars to check SL/TP/EOD."""
    still_open = []
    eod_force = now_dt.time() >= EOD_FORCE_CLOSE
    for pos in state["open_positions"]:
        sym = pos["symbol"]
        df = candle_cache.get(sym)
        if df is None or df.empty:
            still_open.append(pos); continue
        entry_dt = pd.to_datetime(pos["entry_dt"])
        # bars AFTER entry
        after = df[df["dt"] > entry_dt]
        closed = False
        for _, b in after.iterrows():
            if eod_force or b["dt"].time() >= EOD_FORCE_CLOSE:
                close_position(state, pos, float(b["open"]), b["dt"].isoformat(), "eod",
                               f"13:25 EOD force-close at {b['dt'].time().isoformat()[:5]} open={b['open']:.2f}")
                closed = True; break
            if b["high"] >= pos["sl_px"]:
                close_position(state, pos, pos["sl_px"], b["dt"].isoformat(), "sl",
                               f"high={b['high']:.2f} ≥ sl_px={pos['sl_px']:.4f} at {b['dt'].time().isoformat()[:5]}")
                closed = True; break
            if b["low"] <= pos["tp_px"]:
                close_position(state, pos, pos["tp_px"], b["dt"].isoformat(), "tp",
                               f"low={b['low']:.2f} ≤ tp_px={pos['tp_px']:.4f} at {b['dt'].time().isoformat()[:5]}")
                closed = True; break
        if not closed and eod_force and not after.empty:
            last = after.iloc[-1]
            close_position(state, pos, float(last["close"]), last["dt"].isoformat(), "eod",
                           f"EOD force-close at last bar {last['dt'].time().isoformat()[:5]} close={last['close']:.2f}")
            closed = True
        if not closed:
            still_open.append(pos)
    state["open_positions"] = still_open


def get_ref_price(day: str, symbol: str) -> float | None:
    p = BLACKLIST_DIR / f"{day}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    t = (data.get("tickers") or {}).get(symbol, {})
    rp = t.get("referencePrice")
    return float(rp) if rp else None


def load_strategy_predictors() -> list[dict]:
    """Return list of strategies. Each: {name, predict_for_feats(feats)→df_with_proba}."""
    strategies = []

    # global6 — original single model
    if MODEL_PATH.exists():
        g_model = joblib.load(MODEL_PATH)
        LIQUID6 = {"6770", "2337", "1802", "2408", "3481", "1815"}
        def _pred_global(feats, _m=g_model, _syms=LIQUID6):
            f = feats[feats["symbol"].isin(_syms)].copy()
            if f.empty:
                return f
            f["proba"] = _m.predict_proba(f[T.FEATURE_COLS].values)[:, 1]
            return f
        strategies.append({"name": "global6", "predict": _pred_global})

    # per_group
    if PER_GROUP_DIR.exists() and any(PER_GROUP_DIR.glob("*.joblib")):
        # symbol → group lookup
        groups = json.loads(STOCKS_JSON.read_text())["groups"]
        sym_to_group: dict[str, str] = {}
        for g, members in groups.items():
            for s in members:
                sym_to_group[s] = g
        group_models = {g.stem: joblib.load(g) for g in PER_GROUP_DIR.glob("*.joblib")}
        def _pred_group(feats, _models=group_models, _lookup=sym_to_group):
            out_chunks = []
            for sym, sub in feats.groupby("symbol"):
                grp = _lookup.get(sym)
                if grp is None or grp not in _models:
                    continue
                sub = sub.copy()
                sub["proba"] = _models[grp].predict_proba(sub[T.FEATURE_COLS].values)[:, 1]
                out_chunks.append(sub)
            return pd.concat(out_chunks, ignore_index=True) if out_chunks else feats.iloc[0:0].copy()
        strategies.append({"name": "per_group", "predict": _pred_group})

    # per_stock
    if PER_STOCK_DIR.exists() and any(PER_STOCK_DIR.glob("*.joblib")):
        stock_models = {p.stem: joblib.load(p) for p in PER_STOCK_DIR.glob("*.joblib")}
        def _pred_stock(feats, _models=stock_models):
            out_chunks = []
            for sym, sub in feats.groupby("symbol"):
                if sym not in _models:
                    continue
                sub = sub.copy()
                sub["proba"] = _models[sym].predict_proba(sub[T.FEATURE_COLS].values)[:, 1]
                out_chunks.append(sub)
            return pd.concat(out_chunks, ignore_index=True) if out_chunks else feats.iloc[0:0].copy()
        strategies.append({"name": "per_stock", "predict": _pred_stock})

    return strategies


def cycle_for_strategy(strategy: dict, day: str, feats: pd.DataFrame,
                       candle_cache: dict[str, pd.DataFrame], blacklist: set[str]) -> None:
    """Run one cycle for ONE strategy: predict → process exits → open new positions → persist."""
    name = strategy["name"]
    state = load_state(day, name)

    # Process exits FIRST (don't risk missing a SL during prediction window)
    process_open_positions(state, candle_cache, _now())

    # Predict
    pred = strategy["predict"](feats)
    if "proba" in pred.columns:
        pred["conf"] = 1 - pred["proba"]
        pred["triggered"] = pred["proba"] <= PROBA_SHORT_THRESHOLD
    else:
        pred["triggered"] = False

    # TAIEX regime gate — if today is strong-trend + gap-up, block all NEW shorts.
    # Open positions still get processed for exits above; only entries are blocked.
    regime_block, regime_info = taiex_regime_gate(day)
    if regime_block and not state.get("taiex_gate_logged"):
        print(f"  [{name}/REGIME] short entries blocked — {regime_info['reason']}", file=sys.stderr)
        state["taiex_gate_logged"] = True

    # Process new triggered signals
    processed_set = {(s, d) for s, d in state["processed_signals"]}
    if not state["breaker_tripped"]:
        for _, s in pred.iterrows():
            if not s.get("triggered", False):
                continue
            sym = str(s["symbol"])
            key = (sym, str(s["datetime"]))
            if key in processed_set:
                continue
            sig_dt = pd.to_datetime(s["datetime"])
            if sig_dt.tz is not None:
                sig_dt = sig_dt.tz_localize(None)
            row_base = {"datetime": str(s["datetime"]), "symbol": sym,
                        "close": float(s["close"]), "conf": round(float(s["conf"]), 3),
                        "proba": round(float(s["proba"]), 4), "triggered": True}
            if regime_block:
                row_base["status"] = "skip_taiex_regime"
                row_base["taiex_trend"] = regime_info["trend"]
                row_base["taiex_gap"] = regime_info["gap"]
                state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                continue
            if sym in blacklist:
                row_base["status"] = "skip_no_short_sell"
                state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                continue
            rp = get_ref_price(day, sym)
            if rp and rp > 0:
                pct = (float(s["close"]) - rp) / rp
                if pct >= MAX_INTRADAY_PCT_FOR_SHORT:
                    row_base["status"] = "skip_pct_high"; row_base["intraday_pct"] = round(pct, 4)
                    state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
            entry_dt_target = sig_dt + pd.Timedelta(minutes=5)
            cdf = candle_cache.get(sym)
            if cdf is None or cdf.empty:
                continue
            after = cdf[cdf["dt"] >= entry_dt_target]
            if after.empty:
                continue
            entry_price = float(after.iloc[0]["open"])
            notional = entry_price * SHARES_PER_LOT
            if state["budget_used"] + notional > DAILY_BUDGET:
                row_base["status"] = "skip_budget"
                state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                continue
            open_position(state, row_base, entry_price, after.iloc[0]["dt"].isoformat())
            print(f"  [{name}/OPEN] {sym} entry={entry_price:.2f} sl={entry_price*1.025:.2f} "
                  f"tp={entry_price*0.97:.2f} conf={s['conf']:.3f}", file=sys.stderr)
            processed_set.add(key)

    save_state(day, state, name)
    write_paper_csv(day, state, name)


def fetch_taiex_intraday(sdk, day: str) -> None:
    """Fetch TAIEX 1m candles for today and write to taiex/intraday_1m/{day}.csv.
    Used by the regime gate. Best-effort: failure logged but doesn't block cycle.
    """
    try:
        r = sdk.marketdata.rest_client.stock.intraday.candles(symbol="IX0001", timeframe="1")
        bars = r.get("data") or []
        if not bars:
            return
        df = pd.DataFrame(bars)
        df["dt"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        cols = ["dt", "open", "high", "low", "close"]
        if "volume" in df.columns:
            cols.append("volume")
        df = df[cols].sort_values("dt").reset_index(drop=True)
        TAIEX_INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TAIEX_INTRADAY_DIR / f"{day}.csv", index=False)
    except Exception as e:
        print(f"  [taiex] ERROR {e}", file=sys.stderr)


def cycle_multi(sdk, strategies: list[dict], day: str, symbols: list[str]) -> None:
    """Multi-strategy cycle: shared feature pipeline, run each strategy independently."""
    blacklist = load_blacklist(day)
    candle_cache: dict[str, pd.DataFrame] = {}

    fetch_taiex_intraday(sdk, day)

    for sym in symbols:
        try:
            df = fetch_candles_for(sdk, sym)
            if not df.empty:
                save_intraday(day, sym, df)
                candle_cache[sym] = df
        except Exception as e:
            print(f"  [poll] {sym} ERROR {e}", file=sys.stderr)

    try:
        feats = build_features_for_today(day, symbols)
    except Exception as e:
        print(f"  [features] ERROR {e}", file=sys.stderr)
        return

    for strategy in strategies:
        try:
            cycle_for_strategy(strategy, day, feats, candle_cache, blacklist)
        except Exception as e:
            print(f"  [{strategy['name']}] ERROR {e}", file=sys.stderr)


# Legacy single-strategy cycle kept for backward-compat; not used by main loop.
def cycle(sdk, model, day: str, symbols: list[str]) -> None:
    state = load_state(day)
    blacklist = load_blacklist(day)
    candle_cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = fetch_candles_for(sdk, sym)
            if not df.empty:
                save_intraday(day, sym, df)
                candle_cache[sym] = df
        except Exception as e:
            print(f"  [poll] {sym} ERROR {e}", file=sys.stderr)
    process_open_positions(state, candle_cache, _now())
    try:
        feats = build_features_for_today(day, symbols)
    except Exception as e:
        print(f"  [features] ERROR {e}", file=sys.stderr)
        save_state(day, state); write_paper_csv(day, state)
        return
    feats = predict(model, feats)

    # 4. Process new triggered signals
    processed_set = {(s, d) for s, d in state["processed_signals"]}
    if not state["breaker_tripped"]:
        for _, s in feats.iterrows():
            if not s.get("triggered", False):
                continue
            key = (str(s["symbol"]), str(s["datetime"]))
            if key in processed_set:
                continue
            sym = str(s["symbol"])
            sig_dt = pd.to_datetime(s["datetime"]).tz_localize(None) if pd.to_datetime(s["datetime"]).tz else pd.to_datetime(s["datetime"])

            row_base = {"datetime": str(s["datetime"]), "symbol": sym,
                        "close": float(s["close"]), "conf": round(float(s["conf"]), 3),
                        "proba": round(float(s["proba"]), 4), "triggered": True}

            # Blacklist filter
            if sym in blacklist:
                row_base["status"] = "skip_no_short_sell"
                state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                continue
            # 7% filter
            rp = get_ref_price(day, sym)
            if rp and rp > 0:
                pct = (float(s["close"]) - rp) / rp
                if pct >= MAX_INTRADAY_PCT_FOR_SHORT:
                    row_base["status"] = "skip_pct_high"
                    row_base["intraday_pct"] = round(pct, 4)
                    state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
            # Need the NEXT 1-min bar after sig_dt+5min to know entry price
            entry_dt_target = sig_dt + pd.Timedelta(minutes=5)
            cdf = candle_cache.get(sym)
            if cdf is None or cdf.empty:
                continue  # try next cycle
            after = cdf[cdf["dt"] >= entry_dt_target]
            if after.empty:
                continue  # not yet available; retry next cycle
            entry_price = float(after.iloc[0]["open"])
            entry_dt_actual = after.iloc[0]["dt"].isoformat()
            notional = entry_price * SHARES_PER_LOT
            if state["budget_used"] + notional > DAILY_BUDGET:
                row_base["status"] = "skip_budget"
                state["skipped"].append(row_base); state["processed_signals"].append([sym, str(s["datetime"])])
                continue
            open_position(state, row_base, entry_price, entry_dt_actual)
            print(f"  [OPEN]  {sym} entry={entry_price:.2f} sl={entry_price*1.025:.2f} tp={entry_price*0.97:.2f} "
                  f"conf={s['conf']:.3f}", file=sys.stderr)
            processed_set.add(key)

    # 5. Persist
    save_state(day, state)
    write_paper_csv(day, state)


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Heuristic: detect 401/token-expired/auth-related SDK errors."""
    msg = (str(exc) + " " + type(exc).__name__).lower()
    return any(t in msg for t in ("401", "unauthorized", "token", "auth", "expired", "forbidden"))


def main() -> int:
    print(f"=== live_paper start @ {_now().isoformat()} ===", file=sys.stderr)
    strategies = load_strategy_predictors()
    if not strategies:
        raise SystemExit("no strategy models found (run train_strategies.py first)")
    print(f"  strategies loaded: {[s['name'] for s in strategies]}", file=sys.stderr)
    sdk_ref = {"sdk": _build_sdk()}
    stop = {"flag": False}

    def _handle_signal(signum, frame):
        print(f"[live_paper] signal {signum} received → exiting", file=sys.stderr)
        stop["flag"] = True
    sigmod.signal(sigmod.SIGTERM, _handle_signal)
    sigmod.signal(sigmod.SIGINT, _handle_signal)

    last_minute = -1
    while not stop["flag"]:
        now = _now()
        if now.time() < MARKET_OPEN:
            print(f"  waiting for market open (now={now.time().isoformat()[:5]})", file=sys.stderr)
            time.sleep(20); continue
        if now.time() >= MARKET_CLOSE:
            print(f"[live_paper] market closed — final cycle then exit", file=sys.stderr)
            try:
                cycle_multi(sdk_ref["sdk"], strategies, _today(), load_universe())
            except Exception as e:
                print(f"[cycle/final] {type(e).__name__}: {e}", file=sys.stderr)
            break
        if now.minute == last_minute:
            time.sleep(2); continue
        last_minute = now.minute

        symbols = load_universe()
        print(f"\n[{now.time().isoformat()[:5]}] cycle  universe={symbols}  strategies={[s['name'] for s in strategies]}", file=sys.stderr)
        try:
            cycle_multi(sdk_ref["sdk"], strategies, _today(), symbols)
        except Exception as e:
            print(f"[cycle] {type(e).__name__}: {e}", file=sys.stderr)
            if _looks_like_auth_error(e):
                print("[cycle] auth-like error → rebuilding SDK", file=sys.stderr)
                try:
                    sdk_ref["sdk"] = _build_sdk(retries=2)
                    cycle_multi(sdk_ref["sdk"], strategies, _today(), symbols)
                    print("[cycle] recovered after SDK rebuild", file=sys.stderr)
                except SystemExit:
                    raise
                except Exception as e2:
                    print(f"[cycle/retry] still failing: {type(e2).__name__}: {e2}", file=sys.stderr)
        # sleep to top of next minute
        sleep_to = (60 - _now().second) + 1
        time.sleep(min(sleep_to, POLL_SEC))

    print(f"=== live_paper stop @ {_now().isoformat()} ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
