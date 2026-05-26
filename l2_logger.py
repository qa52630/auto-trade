"""Fugle WebSocket L2 (best 5 quotes) + tick logger via Taishin SDK auth.

Logs raw messages from the `books` and `trades` channels to JSONL files under
data/l2/YYYY-MM-DD/<channel>.jsonl. Each line is one server message verbatim
plus a `recv_ts` field (local receive time, ms epoch).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path

from taishin_sdk import TaishinSDK


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).parent
OUT_ROOT = ROOT / "data" / "l2"
LIQUID_SYMBOLS = ["6770", "2337", "1802", "2408", "3481", "1815"]


def login_and_get_token():
    load_env(ROOT / ".env")
    user = os.environ["API_USER"]
    pwd = os.environ["API_PASS"]
    cert_pass = os.environ["API_CERT_PASS"]
    pfx = next(ROOT.glob("*.pfx"))
    api_url = os.environ.get("TAISHIN_API_URL", "https://fugletrade.tssco.com.tw")
    print(f"[auth] login user={user} cert={pfx.name} api_url={api_url}")
    sdk = TaishinSDK(api_url=api_url)
    accounts = sdk.login(user, pwd, str(pfx), cert_pass)
    if not accounts:
        raise SystemExit("login returned no accounts")
    account = accounts[0]
    print(f"[auth] account={account}")
    sdk.init_realtime(account)
    token_obj = sdk.get_realtime_token(account)
    token = token_obj.realtime_token if hasattr(token_obj, "realtime_token") else str(token_obj)
    print(f"[auth] got realtime token ({len(token)} chars)")
    return sdk


def _open_writer(channel: str):
    today = datetime.now().strftime("%Y-%m-%d")
    d = OUT_ROOT / today
    d.mkdir(parents=True, exist_ok=True)
    return (d / f"{channel}.jsonl").open("a", buffering=1)


def main():
    sdk = login_and_get_token()
    # use the marketdata client that init_realtime built (correct URL + token)
    stock = sdk.marketdata.websocket_client.stock

    writers = {"books": _open_writer("books"), "trades": _open_writer("trades")}
    counts = {"books": 0, "trades": 0, "other": 0, "errors": 0}
    start = time.time()

    def on_message(message):
        try:
            msg = json.loads(message) if isinstance(message, (str, bytes)) else message
            ch = msg.get("channel", "other")
            msg["recv_ts"] = int(time.time() * 1000)
            w = writers.get(ch)
            if w:
                w.write(json.dumps(msg, ensure_ascii=False) + "\n")
                counts[ch] += 1
            else:
                counts["other"] += 1
            if sum(counts.values()) % 200 == 0:
                elapsed = time.time() - start
                print(f"[recv] books={counts['books']} trades={counts['trades']} "
                      f"other={counts['other']} errors={counts['errors']} "
                      f"elapsed={elapsed:.0f}s")
        except Exception as e:
            counts["errors"] += 1
            print(f"[recv-error] {e}: {message[:200] if isinstance(message,(str,bytes)) else message}")

    def on_connect():
        print(f"[ws] connected → subscribing books+trades for {LIQUID_SYMBOLS}")

    def on_authenticated(_):
        print("[ws] authenticated, sending subscriptions")
        stock.subscribe({"channel": "books", "symbols": LIQUID_SYMBOLS})
        stock.subscribe({"channel": "trades", "symbols": LIQUID_SYMBOLS})

    def on_disconnect(code, reason):
        print(f"[ws] disconnected code={code} reason={reason}")

    def on_error(err):
        counts["errors"] += 1
        print(f"[ws] error: {err}")

    stock.on("connect", on_connect)
    stock.on("authenticated", on_authenticated)
    stock.on("message", on_message)
    stock.on("disconnect", on_disconnect)
    stock.on("error", on_error)

    stopping = False
    def shutdown(*_):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\n[stop] disconnecting ...")
        try:
            stock.disconnect()
        except Exception:
            pass
        for w in writers.values():
            try: w.close()
            except Exception: pass
        elapsed = time.time() - start
        print(f"[stop] final counts {counts}  elapsed={elapsed:.0f}s")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[ws] connecting via SDK-managed marketdata client")
    stock.connect()

    # Optional auto-stop at HH:MM local time (e.g., STOP_AT_HHMM=14:00)
    stop_at = None
    stop_env = os.environ.get("STOP_AT_HHMM", "").strip()
    if stop_env and ":" in stop_env:
        hh, mm = stop_env.split(":")
        stop_at = dtime(int(hh), int(mm))
        print(f"[stop] auto-exit scheduled at {stop_at}")

    while not stopping:
        time.sleep(60)
        now = datetime.now()
        print(f"[heartbeat] {now:%H:%M:%S} books={counts['books']} "
              f"trades={counts['trades']} errors={counts['errors']}")
        if stop_at and now.time() >= stop_at:
            print(f"[stop] reached {stop_at}, shutting down")
            shutdown()


if __name__ == "__main__":
    main()
