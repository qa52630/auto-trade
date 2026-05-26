"""Realtime paper trader — WebSocket tick-driven (replaces 60s REST polling).

Key changes vs. live_paper.py:
  - **Tick-level SL/TP**: every trade tick checked against open positions; closes
    fire at the actual tick price (not assumed-limit fill), reflecting real slippage.
  - **Bar prediction still REST**: completed 1-min candles fetched via REST every
    minute → 5-min features → model predict. Predictions only fire on bar close,
    so REST is the right level for that.
  - **Entry at next tick**: when a signal fires, the NEXT incoming trade tick
    for that symbol becomes the simulated entry price (instead of next-bar open).
    Mimics market-order behaviour.

Threading:
  - Main thread: REST cycle loop (1-min cadence)
  - WS callback thread: tick handler, SL/TP monitor, entry filler
  - shared `state` protected by a single Lock

Output files match live_paper.py:
  data/paper_trades/{day}.csv                  (global6)
  data/paper_trades/{day}_per_group.csv
  data/paper_trades/{day}_per_stock.csv
  data/paper_trades/state_{day}*.json
"""
from __future__ import annotations

import json
import os
import signal as sigmod
import sys
import threading
import time
from datetime import datetime, time as dtime, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd

import train as T
import live_paper as LP   # reuse helpers (build_features_for_today, save_state, etc.)

ROOT = LP.ROOT
TZ = LP.TZ
MARKET_OPEN = LP.MARKET_OPEN
MARKET_CLOSE = LP.MARKET_CLOSE
EOD_FORCE_CLOSE = LP.EOD_FORCE_CLOSE
PROBA_SHORT_THRESHOLD = LP.PROBA_SHORT_THRESHOLD
SHORT_SL = LP.SHORT_SL
SHORT_TP = LP.SHORT_TP
COST_RATE = LP.COST_RATE
SHARES_PER_LOT = LP.SHARES_PER_LOT
DAILY_BUDGET = LP.DAILY_BUDGET
DAILY_LOSS_CAP = LP.DAILY_LOSS_CAP
MAX_INTRADAY_PCT_FOR_SHORT = LP.MAX_INTRADAY_PCT_FOR_SHORT

state_lock = threading.Lock()

# Last-seen tick price per symbol (updated by on_tick_message). Used by the
# EOD-force-close watchdog when a symbol has no recent tick after 13:25.
last_tick_price: dict[str, tuple[float, pd.Timestamp]] = {}


def _now() -> datetime:
    return datetime.now(TZ)


def _close_at_tick(state: dict, pos: dict, exit_price: float, exit_dt: str,
                   reason: str, exit_detail: str) -> None:
    """Tick-driven close. Uses the ACTUAL tick price as exit price (no limit fill assumption)."""
    LP.close_position(state, pos, exit_price, exit_dt, reason, exit_detail)


def _check_position_against_tick(pos: dict, tick_price: float, tick_dt: pd.Timestamp,
                                 force_eod: bool) -> tuple[str, str] | None:
    """Decide if this tick triggers an exit. Returns (reason, detail) or None.
    Note: returns actual tick price as exit (not sl_px/tp_px) — that's the realism upgrade."""
    if force_eod:
        return ("eod", f"13:25 EOD force-close at tick {tick_dt.time().isoformat(timespec='seconds')} price={tick_price}")
    if tick_price >= pos["sl_px"]:
        slip = tick_price - pos["sl_px"]
        return ("sl", f"tick={tick_price} ≥ sl_px={pos['sl_px']:.4f} (slip +{slip:.2f}) at {tick_dt.time().isoformat(timespec='seconds')}")
    if tick_price <= pos["tp_px"]:
        slip = pos["tp_px"] - tick_price
        return ("tp", f"tick={tick_price} ≤ tp_px={pos['tp_px']:.4f} (slip +{slip:.2f}) at {tick_dt.time().isoformat(timespec='seconds')}")
    return None


PENDING_TIMEOUT_SEC = 300   # signals waiting > 5 min get dropped (低流動性保護)


def _has_open_position(state: dict, sym: str) -> bool:
    return any(p["symbol"] == sym for p in state.get("open_positions", []))


def _process_pending_entries(state: dict, sym: str, tick_price: float, tick_dt: pd.Timestamp,
                             day: str, strategy_name: str) -> None:
    """If state has pending entry orders for this symbol, fill them at current tick.
    Skips if same symbol already has open position (no duplicate stacking)."""
    pending = state.get("pending_entries", [])
    still_pending = []
    for sig in pending:
        if sig["symbol"] != sym:
            still_pending.append(sig); continue
        # #1 fix: block duplicate-symbol stacking
        if _has_open_position(state, sym):
            sig["status"] = "skip_already_open"
            state["skipped"].append(sig); state["processed_signals"].append([sym, sig["datetime"]])
            continue
        # Timeout drop
        queued_at = sig.get("queued_at")
        if queued_at is not None and (time.time() - queued_at) > PENDING_TIMEOUT_SEC:
            sig["status"] = "skip_pending_timeout"
            state["skipped"].append(sig); state["processed_signals"].append([sym, sig["datetime"]])
            continue
        entry_price = float(tick_price)
        notional = entry_price * SHARES_PER_LOT
        if state["budget_used"] + notional > DAILY_BUDGET:
            sig["status"] = "skip_budget_at_fill"
            state["skipped"].append(sig); state["processed_signals"].append([sym, sig["datetime"]])
            continue
        LP.open_position(state, sig, entry_price, tick_dt.isoformat())
        state["open_positions"][-1]["fill_via"] = "tick"
        print(f"  [{strategy_name}/OPEN-tick] {sym} entry={entry_price:.2f} "
              f"sl={entry_price*1.025:.2f} tp={entry_price*0.97:.2f} "
              f"conf={sig['conf']:.3f}", file=sys.stderr)
    state["pending_entries"] = still_pending


def on_tick_message(msg: dict, strategies_state: dict, day: str) -> None:
    """Handle one trade tick. Closes SL/TP-triggered positions; fills pending entries."""
    d = msg.get("data") or {}
    sym = str(d.get("symbol", ""))
    if not sym:
        return
    raw_t = d.get("time")
    if raw_t is None or not d.get("price"):
        return
    tick_price = float(d["price"])
    # epoch microseconds → datetime
    tick_dt = pd.to_datetime(int(raw_t), unit="us").tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    eod = tick_dt.time() >= EOD_FORCE_CLOSE
    last_tick_price[sym] = (tick_price, tick_dt)

    with state_lock:
        for strat_name, state in strategies_state.items():
            # 1. Check open positions against this tick
            still_open = []
            for pos in state["open_positions"]:
                if pos["symbol"] != sym:
                    still_open.append(pos); continue
                # don't exit on same-bar entry tick (need >= 1 future tick)
                entry_dt = pd.to_datetime(pos["entry_dt"])
                if entry_dt.tz is not None: entry_dt = entry_dt.tz_localize(None)
                if tick_dt <= entry_dt:
                    still_open.append(pos); continue
                trig = _check_position_against_tick(pos, tick_price, tick_dt, force_eod=eod)
                if trig is None:
                    still_open.append(pos)
                else:
                    reason, detail = trig
                    _close_at_tick(state, pos, tick_price, tick_dt.isoformat(), reason, detail)
            state["open_positions"] = still_open
            # 2. Fill any pending entries for this symbol
            if state.get("pending_entries"):
                _process_pending_entries(state, sym, tick_price, tick_dt, day, strat_name)


def cycle_predict(sdk, strategies: list[dict], strategies_state: dict, day: str,
                  symbols: list[str]) -> None:
    """REST cycle: poll candles → predict → ENQUEUE entries (filled by next tick)."""
    blacklist = LP.load_blacklist(day)
    candle_cache: dict[str, pd.DataFrame] = {}

    LP.fetch_taiex_intraday(sdk, day)

    for sym in symbols:
        try:
            df = LP.fetch_candles_for(sdk, sym)
            if not df.empty:
                LP.save_intraday(day, sym, df)
                candle_cache[sym] = df
        except Exception as e:
            print(f"  [poll] {sym} ERROR {e}", file=sys.stderr)

    try:
        feats = LP.build_features_for_today(day, symbols)
    except Exception as e:
        print(f"  [features] ERROR {e}", file=sys.stderr)
        return

    regime_block, regime_info = LP.taiex_regime_gate(day)

    with state_lock:
        for strategy in strategies:
            name = strategy["name"]
            state = strategies_state[name]
            if state.get("breaker_tripped"):
                continue
            if regime_block and not state.get("taiex_gate_logged"):
                print(f"  [{name}/REGIME] short entries blocked — {regime_info['reason']}",
                      file=sys.stderr)
                state["taiex_gate_logged"] = True
            pred = strategy["predict"](feats)
            if "proba" not in pred.columns or pred.empty:
                continue
            pred["conf"] = 1 - pred["proba"]
            pred["triggered"] = pred["proba"] <= PROBA_SHORT_THRESHOLD

            processed = {(s, d) for s, d in state["processed_signals"]}
            for _, s in pred.iterrows():
                if not s.get("triggered", False):
                    continue
                sym = str(s["symbol"])
                key = (sym, str(s["datetime"]))
                if key in processed:
                    continue
                row_base = {"datetime": str(s["datetime"]), "symbol": sym,
                            "close": float(s["close"]), "conf": round(float(s["conf"]), 3),
                            "proba": round(float(s["proba"]), 4), "triggered": True}
                if regime_block:
                    row_base["status"] = "skip_taiex_regime"
                    row_base["taiex_trend"] = regime_info["trend"]
                    row_base["taiex_gap"] = regime_info["gap"]
                    state["skipped"].append(row_base)
                    state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
                if sym in blacklist:
                    row_base["status"] = "skip_no_short_sell"
                    state["skipped"].append(row_base)
                    state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
                rp = LP.get_ref_price(day, sym)
                if rp and rp > 0:
                    pct = (float(s["close"]) - rp) / rp
                    if pct >= MAX_INTRADAY_PCT_FOR_SHORT:
                        row_base["status"] = "skip_pct_high"
                        row_base["intraday_pct"] = round(pct, 4)
                        state["skipped"].append(row_base)
                        state["processed_signals"].append([sym, str(s["datetime"])])
                        continue
                # #1 fix: skip if already holding or queued for this symbol
                if _has_open_position(state, sym):
                    row_base["status"] = "skip_already_open"
                    state["skipped"].append(row_base)
                    state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
                if any(q["symbol"] == sym for q in state.get("pending_entries", [])):
                    row_base["status"] = "skip_already_queued"
                    state["skipped"].append(row_base)
                    state["processed_signals"].append([sym, str(s["datetime"])])
                    continue
                # Queue: next tick for this symbol → fill
                row_base["queued_at"] = time.time()
                state.setdefault("pending_entries", []).append(row_base)
                processed.add(key)
                print(f"  [{name}/queue] {sym} signal at {s['datetime']} "
                      f"conf={s['conf']:.3f} — awaiting next tick", file=sys.stderr)

            # persist after enqueue
            LP.save_state(day, state, name)
            LP.write_paper_csv(day, state, name)


def eod_force_close_sweep(strategies_state: dict, day: str) -> int:
    """#6 fix: at/after 13:25, force-close any open position that hasn't seen a tick.
    Uses last_tick_price cache; falls back to entry_price if even that's missing.
    Returns count of positions force-closed."""
    now = _now()
    if now.time() < EOD_FORCE_CLOSE:
        return 0
    closed_count = 0
    with state_lock:
        for name, state in strategies_state.items():
            still_open = []
            for pos in state["open_positions"]:
                sym = pos["symbol"]
                # Use last tick if available, else last-known close from intraday
                fallback = last_tick_price.get(sym)
                if fallback is not None:
                    exit_price, exit_dt = float(fallback[0]), fallback[1]
                else:
                    # Fall back to entry price (no tick activity at all)
                    exit_price = float(pos["entry_price"])
                    exit_dt = now.replace(tzinfo=None)
                LP.close_position(state, pos, exit_price, exit_dt.isoformat(), "eod",
                                  f"EOD watchdog sweep at {now.time().isoformat(timespec='seconds')} "
                                  f"(no tick since 13:25, used last_tick={exit_price})")
                closed_count += 1
            state["open_positions"] = still_open
            if closed_count:
                LP.save_state(day, state, name)
                LP.write_paper_csv(day, state, name)
    return closed_count


def state_flush_loop(strategies_state: dict, day: str, stop: dict) -> None:
    """Background thread: persist state every 10s during market hours."""
    while not stop["flag"]:
        time.sleep(10)
        try:
            with state_lock:
                for name, state in strategies_state.items():
                    LP.save_state(day, state, name)
                    LP.write_paper_csv(day, state, name)
        except Exception as e:
            # Never let a transient persistence failure kill the flush thread.
            print(f"  [flush] ERROR {e!r} — will retry next cycle", file=sys.stderr)


def _next_trading_open(now: datetime) -> datetime:
    """Return datetime of the next market open, skipping weekends.
    If now is during today's market hours, returns today's 09:00 (in the past)."""
    target = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                         second=0, microsecond=0)
    # Advance to next day if past today's close, or if today is weekend
    if now.time() >= MARKET_CLOSE or now.weekday() >= 5:
        target = target + timedelta(days=1)
    while target.weekday() >= 5:
        target = target + timedelta(days=1)
    return target


def _wait_until_next_open(stop: dict) -> bool:
    """Sleep until the next market open. Returns True if reached the open,
    False if interrupted by stop flag."""
    next_open = _next_trading_open(_now())
    print(f"[wait] sleeping until next market open: {next_open.isoformat(timespec='seconds')} "
          f"({_now().strftime('%A')} now)", file=sys.stderr)
    while not stop["flag"]:
        now = _now()
        remaining = (next_open - now).total_seconds()
        if remaining <= 0:
            return True
        # Sleep in 60s chunks so SIGTERM responsive
        time.sleep(min(remaining, 60))
    return False


def main() -> int:
    print(f"=== live_paper_realtime start @ {_now().isoformat()} ===", file=sys.stderr)
    strategies = LP.load_strategy_predictors()
    if not strategies:
        raise SystemExit("no strategy models found")
    print(f"  strategies: {[s['name'] for s in strategies]}", file=sys.stderr)

    stop = {"flag": False}

    def sigterm(*_):
        stop["flag"] = True
        print("[live_paper_realtime] SIGTERM/SIGINT → exiting", file=sys.stderr)
    sigmod.signal(sigmod.SIGTERM, sigterm)
    sigmod.signal(sigmod.SIGINT, sigterm)

    # Outer loop: each iteration handles one trading day. After close (or weekend),
    # sleep until the next trading-day open and start over.
    while not stop["flag"]:
        now = _now()
        # If we're outside this day's trading window (weekend, after close, or
        # well before open), wait until the next session.
        if now.weekday() >= 5 or now.time() >= MARKET_CLOSE:
            if not _wait_until_next_open(stop):
                break
            continue

        day = _now().date().isoformat()
        print(f"\n=== trading session {day} ({_now().strftime('%A')}) ===", file=sys.stderr)
        strategies_state = {s["name"]: LP.load_state(day, s["name"]) for s in strategies}
        for st in strategies_state.values():
            st.setdefault("pending_entries", [])

        sdk = LP._build_sdk()

        # Wait until market open
        while not stop["flag"] and _now().time() < MARKET_OPEN:
            print(f"  waiting for market open (now={_now().time().isoformat(timespec='seconds')})",
                  file=sys.stderr)
            time.sleep(20)
        if stop["flag"]:
            break

        universe = LP.load_universe()
        print(f"  universe: {universe}", file=sys.stderr)

        # ---- Per-trading-day setup: WS connection, handlers, flusher ----
        ws = {"sdk": sdk, "stock": sdk.marketdata.websocket_client.stock,
              "connected": False, "reconnect_at": None, "fail_count": 0,
              "subscribed_universe": None, "last_tick_ts": None}
        RECONNECT_BACKOFF = [5, 10, 30, 60, 120, 300]
        MAX_TICK_SILENCE = 180

        def _attach_handlers(stock_client, universe):
            def on_message(message):
                try:
                    msg = json.loads(message) if isinstance(message, (str, bytes)) else message
                    if msg.get("channel") != "trades":
                        return
                    ws["last_tick_ts"] = time.time()
                    on_tick_message(msg, strategies_state, day)
                except Exception as e:
                    print(f"[ws] handler error {type(e).__name__}: {e}", file=sys.stderr)

            def on_authenticated(_):
                ws["connected"] = True
                ws["fail_count"] = 0
                ws["last_tick_ts"] = time.time()
                print(f"[ws] authenticated → subscribing trades for {universe}", file=sys.stderr)
                try:
                    stock_client.subscribe({"channel": "trades", "symbols": universe})
                    ws["subscribed_universe"] = list(universe)
                except Exception as e:
                    print(f"[ws] subscribe error: {e}", file=sys.stderr)

            def on_disconnect(code, reason):
                was = ws["connected"]
                ws["connected"] = False
                if stop["flag"]:
                    print(f"[ws] disconnect during shutdown — OK (code={code})", file=sys.stderr)
                    return
                ws["fail_count"] += 1
                backoff = RECONNECT_BACKOFF[min(ws["fail_count"]-1, len(RECONNECT_BACKOFF)-1)]
                ws["reconnect_at"] = time.time() + backoff
                print(f"[ws] DISCONNECTED code={code} reason={reason}  was_connected={was}  "
                      f"reconnect in {backoff}s (attempt {ws['fail_count']})", file=sys.stderr)

            def on_error(err):
                print(f"[ws] error: {err}", file=sys.stderr)

            stock_client.on("authenticated", on_authenticated)
            stock_client.on("message", on_message)
            stock_client.on("disconnect", on_disconnect)
            stock_client.on("error", on_error)

        def _connect_ws(universe):
            try:
                _attach_handlers(ws["stock"], universe)
                ws["stock"].connect()
                return True
            except Exception as e:
                print(f"[ws] connect failed: {e}", file=sys.stderr)
                ws["fail_count"] += 1
                backoff = RECONNECT_BACKOFF[min(ws["fail_count"]-1, len(RECONNECT_BACKOFF)-1)]
                ws["reconnect_at"] = time.time() + backoff
                return False

        def _reconnect_ws():
            print(f"[ws] reconnecting... (fail_count={ws['fail_count']})", file=sys.stderr)
            try:
                try:
                    ws["stock"].disconnect()
                except Exception:
                    pass
                new_sdk = LP._build_sdk(retries=2)
                ws["sdk"] = new_sdk
                ws["stock"] = new_sdk.marketdata.websocket_client.stock
                ws["reconnect_at"] = None
                return _connect_ws(LP.load_universe())
            except SystemExit:
                raise
            except Exception as e:
                print(f"[ws] reconnect failed: {e}", file=sys.stderr)
                ws["fail_count"] += 1
                backoff = RECONNECT_BACKOFF[min(ws["fail_count"]-1, len(RECONNECT_BACKOFF)-1)]
                ws["reconnect_at"] = time.time() + backoff
                return False

        _connect_ws(universe)

        # State flusher thread (one per session)
        session_stop = {"flag": False}
        flusher = threading.Thread(target=state_flush_loop,
                                   args=(strategies_state, day, session_stop), daemon=True)
        flusher.start()

        # Inner loop — runs until market close or shutdown
        last_minute = -1
        while not stop["flag"]:
            now = _now()
            if now.time() >= MARKET_CLOSE:
                print(f"[session {day}] market closed — final flush", file=sys.stderr)
                with state_lock:
                    for name, st in strategies_state.items():
                        LP.save_state(day, st, name)
                        LP.write_paper_csv(day, st, name)
                break

            if now.time() >= EOD_FORCE_CLOSE:
                n = eod_force_close_sweep(strategies_state, day)
                if n:
                    print(f"[eod-sweep] force-closed {n} positions @ "
                          f"{now.time().isoformat(timespec='seconds')}", file=sys.stderr)

            if not ws["connected"] and ws["reconnect_at"] and time.time() >= ws["reconnect_at"]:
                _reconnect_ws()

            if ws["connected"] and ws["last_tick_ts"] and now.time() >= MARKET_OPEN:
                silence = time.time() - ws["last_tick_ts"]
                if silence > MAX_TICK_SILENCE:
                    print(f"[ws] WATCHDOG: {silence:.0f}s silence → forcing reconnect",
                          file=sys.stderr)
                    ws["connected"] = False
                    ws["fail_count"] += 1
                    ws["reconnect_at"] = time.time()

            if now.minute != last_minute:
                last_minute = now.minute
                print(f"\n[{now.time().isoformat(timespec='seconds')[:5]}] REST cycle "
                      f"(ws_connected={ws['connected']} fail={ws['fail_count']})", file=sys.stderr)
                try:
                    cycle_predict(ws["sdk"], strategies, strategies_state, day, LP.load_universe())
                except Exception as e:
                    print(f"[cycle] {type(e).__name__}: {e}", file=sys.stderr)
                    if LP._looks_like_auth_error(e):
                        try:
                            ws["sdk"] = LP._build_sdk(retries=2)
                        except SystemExit:
                            raise
            time.sleep(3)

        # End of session: disconnect WS + stop flusher; outer loop sleeps to next open
        session_stop["flag"] = True
        try:
            ws["stock"].disconnect()
        except Exception:
            pass
        time.sleep(2)   # let flusher thread observe stop flag and exit

    print(f"=== live_paper_realtime stop @ {_now().isoformat()} ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
