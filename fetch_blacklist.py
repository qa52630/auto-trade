"""Fetch per-symbol trading status from Fugle intraday/ticker API.

For each symbol in our universe, queries:
  - isDisposition       (處置股)
  - canDayTrade         (可現沖整體開關)
  - canBuyDayTrade      (可先買後賣現沖; 若 false 代表「暫停先買後賣」)
  - referencePrice      (參考價, 用於 7% 漲幅過濾)
  - limitUpPrice / limitDownPrice
  - matchingInterval    (撮合間隔; 處置股 = 300s)

Output: data/blacklist/YYYY-MM-DD.json
  {
    "day": "2026-05-22",
    "tickers": {"2337": {...full ticker dict...}, ...},
    "disposition":     ["1809", ...],   # isDisposition = true
    "no_day_trade":    [...],            # canDayTrade   = false
    "no_short_sell":   [...],            # short-side day-trade blocked
    "fetched_at": "ISO"
  }

「不可放空」的判定:
  - isDisposition = true (處置股一律排除)
  - 或 canDayTrade = false (整體不可現沖)
  注意: 「暫停先賣後買」TWSE 公告為主, Fugle API 沒有獨立欄位。
        canBuyDayTrade=false 代表「暫停先買後賣」(這個對放空策略不重要)。

Run:
  python3 fetch_blacklist.py                 # 抓 universe 各檔今日狀態
  python3 fetch_blacklist.py 2337 6770       # 只查特定 symbols
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "blacklist"
TZ = timezone(timedelta(hours=8))

# Universe (matches train.py LIQUID_SYMBOLS)
DEFAULT_SYMBOLS = ["6770", "2337", "1802", "2408", "3481", "1815"]


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _build_sdk():
    _load_env()
    from taishin_sdk import TaishinSDK
    api_url = os.environ.get("TAISHIN_API_URL", "https://fugletrade.tssco.com.tw")
    sdk = TaishinSDK(api_url=api_url)
    pfx = next(ROOT.glob("*.pfx"))
    accounts = sdk.login(
        os.environ["API_USER"], os.environ["API_PASS"], str(pfx), os.environ["API_CERT_PASS"]
    )
    if not accounts:
        raise SystemExit("login returned no accounts")
    sdk.init_realtime(accounts[0])
    return sdk


import time

MAX_RETRY = 3
RETRY_SLEEP = 2.0


def fetch_tickers(symbols: list[str]) -> dict[str, dict]:
    sdk = _build_sdk()
    out = {}
    for sym in symbols:
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                r = sdk.marketdata.rest_client.stock.intraday.ticker(symbol=sym)
                if hasattr(r, "__dict__") and not isinstance(r, dict):
                    r = vars(r)
                out[sym] = r
                print(
                    f"  {sym} {r.get('name','?')}: "
                    f"disposition={r.get('isDisposition')}  "
                    f"canDayTrade={r.get('canDayTrade')}  "
                    f"canBuyDayTrade={r.get('canBuyDayTrade')}  "
                    f"refPrice={r.get('referencePrice')}"
                    f"{f' (attempt {attempt})' if attempt > 1 else ''}",
                    file=sys.stderr,
                )
                break
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRY:
                    print(f"  {sym}: attempt {attempt} failed ({type(e).__name__}); retrying...", file=sys.stderr)
                    time.sleep(RETRY_SLEEP)
        else:
            print(f"  {sym}: ERROR after {MAX_RETRY} attempts: {last_err}", file=sys.stderr)
            out[sym] = {"error": str(last_err)}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="*", help="symbol codes; default = liquid universe")
    p.add_argument("--day", default=None, help="YYYY-MM-DD; default = today (file naming only)")
    args = p.parse_args()

    day = args.day or datetime.now(TZ).date().isoformat()
    symbols = args.symbols or DEFAULT_SYMBOLS
    print(f"[fetch_blacklist] {day}  symbols={symbols}", file=sys.stderr)

    tickers = fetch_tickers(symbols)

    disposition = sorted(s for s, t in tickers.items() if t.get("isDisposition") is True)
    no_day_trade = sorted(s for s, t in tickers.items() if t.get("canDayTrade") is False)
    # Short-side day-trade blocked = disposition OR canDayTrade=false (no public field
    # specifically gates 先賣後買; both these conditions force the broker to reject it).
    no_short_sell = sorted(set(disposition) | set(no_day_trade))

    print(f"\n  ⚠️ disposition (處置): {disposition}", file=sys.stderr)
    print(f"  ⚠️ no_day_trade: {no_day_trade}", file=sys.stderr)
    print(f"  ⚠️ no_short_sell (final): {no_short_sell}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{day}.json"
    payload = {
        "day": day,
        "tickers": tickers,
        "disposition": disposition,
        "no_day_trade": no_day_trade,
        "no_short_sell": no_short_sell,
        "source": "fugle_intraday_ticker",
        "fetched_at": datetime.now(TZ).isoformat(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n  → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
