"""Build 5-minute L2 / tick features from the day's JSONL captures.

Reads data/l2/YYYY-MM-DD/{books,trades}.jsonl and emits one CSV row per
(symbol, 5-min bucket) with order-book and tick aggregates:

  obi_mean        — mean order book imbalance (bid-ask size) / (bid+ask)
  obi_last        — last bar's OBI
  spread_bps_mean — mean bid-ask spread in basis points of mid
  quote_intensity — count of book updates in the bucket
  trade_count     — trade prints in the bucket
  volume          — sum of trade sizes
  vwap            — volume-weighted average trade price
  aggressor_ratio — buy-initiated volume / total volume (proxy: price>=ask)
  vwap_dev        — (last_trade_price - vwap) / vwap
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date as _date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
L2_ROOT = ROOT / "data" / "l2"
OUT_ROOT = ROOT / "data" / "l2_features"

TZ = timezone(timedelta(hours=8))  # Asia/Taipei

# State accumulated per (symbol, 5-min bucket) for books channel
def _new_book_bucket():
    return {
        "obi_sum": 0.0, "obi_count": 0, "obi_last": None,
        "spread_bps_sum": 0.0, "spread_bps_count": 0,
        "updates": 0,
    }

def _new_trade_bucket():
    return {
        "count": 0, "volume": 0.0,
        "px_vol_sum": 0.0,            # price * size, for vwap
        "buy_vol": 0.0,               # size where price >= last_ask
        "last_price": None, "last_ask": None,
    }


def _bucket_dt(epoch_us: int) -> datetime:
    """Floor microsecond epoch timestamp to the 5-minute bucket start (local TZ)."""
    dt = datetime.fromtimestamp(epoch_us / 1_000_000, tz=timezone.utc).astimezone(TZ)
    floored = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
    return floored


def process_day(day: str) -> Path:
    day_dir = L2_ROOT / day
    books_path = day_dir / "books.jsonl"
    trades_path = day_dir / "trades.jsonl"
    if not books_path.exists() or not trades_path.exists():
        raise SystemExit(f"missing books/trades for {day}: {day_dir}")

    books_state: dict[tuple[str, datetime], dict] = defaultdict(_new_book_bucket)
    trades_state: dict[tuple[str, datetime], dict] = defaultdict(_new_trade_bucket)
    last_ask_by_symbol: dict[str, float] = {}  # rolling last ask for aggressor classification

    # --- books pass ---
    with books_path.open() as f:
        for line in f:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = msg.get("data") or {}
            sym = d.get("symbol")
            t = d.get("time")
            bids = d.get("bids") or []
            asks = d.get("asks") or []
            if not sym or t is None or not bids or not asks:
                continue
            bucket = _bucket_dt(int(t))
            key = (sym, bucket)
            b = books_state[key]
            # OBI from top-of-book
            bid_sz = float(bids[0].get("size", 0))
            ask_sz = float(asks[0].get("size", 0))
            denom = bid_sz + ask_sz
            if denom > 0:
                obi = (bid_sz - ask_sz) / denom
                b["obi_sum"] += obi
                b["obi_count"] += 1
                b["obi_last"] = obi
            # spread in bps
            bid_px = float(bids[0].get("price", 0))
            ask_px = float(asks[0].get("price", 0))
            mid = (bid_px + ask_px) / 2.0
            if mid > 0 and ask_px > bid_px:
                b["spread_bps_sum"] += (ask_px - bid_px) / mid * 10000.0
                b["spread_bps_count"] += 1
            b["updates"] += 1
            last_ask_by_symbol[sym] = ask_px or last_ask_by_symbol.get(sym, 0.0)

    # --- trades pass ---
    with trades_path.open() as f:
        for line in f:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = msg.get("data") or {}
            sym = d.get("symbol")
            t = d.get("time")
            px = d.get("price")
            sz = d.get("size")
            ask = d.get("ask")
            if not sym or t is None or px is None or sz is None:
                continue
            bucket = _bucket_dt(int(t))
            key = (sym, bucket)
            tr = trades_state[key]
            tr["count"] += 1
            tr["volume"] += float(sz)
            tr["px_vol_sum"] += float(px) * float(sz)
            # aggressor: trade price hits the ask → buy-initiated
            cmp_ask = float(ask) if ask is not None else last_ask_by_symbol.get(sym)
            if cmp_ask is not None and float(px) >= cmp_ask:
                tr["buy_vol"] += float(sz)
            tr["last_price"] = float(px)
            tr["last_ask"] = float(ask) if ask is not None else tr["last_ask"]

    # --- merge into rows ---
    all_keys = set(books_state) | set(trades_state)
    rows = []
    for sym, bucket in sorted(all_keys, key=lambda k: (k[0], k[1])):
        b = books_state.get((sym, bucket), _new_book_bucket())
        tr = trades_state.get((sym, bucket), _new_trade_bucket())
        obi_mean = b["obi_sum"] / b["obi_count"] if b["obi_count"] else None
        spread_mean = b["spread_bps_sum"] / b["spread_bps_count"] if b["spread_bps_count"] else None
        vwap = tr["px_vol_sum"] / tr["volume"] if tr["volume"] else None
        aggressor = tr["buy_vol"] / tr["volume"] if tr["volume"] else None
        vwap_dev = ((tr["last_price"] - vwap) / vwap) if (vwap and tr["last_price"] is not None) else None
        rows.append({
            "symbol": sym,
            "datetime": bucket.isoformat(),
            "date": bucket.date().isoformat(),
            "obi_mean": obi_mean,
            "obi_last": b["obi_last"],
            "spread_bps_mean": spread_mean,
            "quote_intensity": b["updates"],
            "trade_count": tr["count"],
            "volume": tr["volume"],
            "vwap": vwap,
            "aggressor_ratio": aggressor,
            "vwap_dev": vwap_dev,
        })

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{day}.csv"
    df.to_csv(out_path, index=False)
    print(f"[l2-features] day={day} buckets={len(df)} symbols={df['symbol'].nunique()} → {out_path}")
    return out_path


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(TZ).date().isoformat()
    process_day(day)


if __name__ == "__main__":
    main()
