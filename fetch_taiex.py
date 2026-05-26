"""Fetch TAIEX (IX0001) intraday 1m and historical daily candles via Taishin SDK.

Outputs:
  data/taiex/intraday_1m/{YYYY-MM-DD}.csv   today's (or --date) 1m bars
  data/taiex/daily.csv                       rolling daily history (merged on date)

Usage:
  python fetch_taiex.py                    # both intraday(today) + daily(last 60)
  python fetch_taiex.py --intraday-only
  python fetch_taiex.py --daily-only --days 250
  python fetch_taiex.py --date 2026-05-25  # intraday for a specific past date

Run inside the Docker container or wherever .env + .pfx are present.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TAIEX_DIR = ROOT / "data" / "taiex"
INTRADAY_DIR = TAIEX_DIR / "intraday_1m"
DAILY_CSV = TAIEX_DIR / "daily.csv"
TZ = timezone(timedelta(hours=8))
TAIEX_SYMBOL = "IX0001"


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def build_sdk():
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


def fetch_intraday(sdk, date: str | None = None) -> pd.DataFrame:
    """Fetch 1-min candles for TAIEX. date=None → today (whatever the API returns)."""
    # Fugle intraday endpoint always returns *today's* bars; date arg is ignored by API
    # but we pass it through for symmetry / future-proofing.
    r = sdk.marketdata.rest_client.stock.intraday.candles(symbol=TAIEX_SYMBOL, timeframe="1")
    bars = r.get("data") or []
    if not bars:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df["dt"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    cols = ["dt", "open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols].sort_values("dt").reset_index(drop=True)


def fetch_daily(sdk, days: int = 60) -> pd.DataFrame:
    """Fetch historical daily candles for TAIEX (last `days` calendar days)."""
    today = datetime.now(TZ).date()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()
    # Fugle endpoint expects keyword `from` (reserved word in Python) — pass via **dict.
    r = sdk.marketdata.rest_client.stock.historical.candles(
        symbol=TAIEX_SYMBOL, timeframe="D",
        **{"from": start, "to": end},
    )
    bars = r.get("data") or []
    if not bars:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    cols = ["date", "open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols].sort_values("date").reset_index(drop=True)


def save_intraday(df: pd.DataFrame, day: str) -> Path:
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    p = INTRADAY_DIR / f"{day}.csv"
    df.to_csv(p, index=False)
    return p


def save_daily(df: pd.DataFrame) -> Path:
    """Merge new rows into existing daily.csv (idempotent on `date`)."""
    TAIEX_DIR.mkdir(parents=True, exist_ok=True)
    if DAILY_CSV.exists():
        old = pd.read_csv(DAILY_CSV)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        merged = df
    merged.to_csv(DAILY_CSV, index=False)
    return DAILY_CSV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intraday-only", action="store_true")
    ap.add_argument("--daily-only", action="store_true")
    ap.add_argument("--days", type=int, default=60, help="daily history lookback (calendar days)")
    ap.add_argument("--date", default=None, help="label for the intraday file (default: today)")
    args = ap.parse_args()

    sdk = build_sdk()

    if not args.daily_only:
        df = fetch_intraday(sdk)
        day = args.date or datetime.now(TZ).date().isoformat()
        if df.empty:
            print(f"[intraday] no bars returned for {TAIEX_SYMBOL}", file=sys.stderr)
        else:
            p = save_intraday(df, day)
            print(f"[intraday] {len(df)} bars  range={df['dt'].min()}..{df['dt'].max()}  → {p}")

    if not args.intraday_only:
        df = fetch_daily(sdk, days=args.days)
        if df.empty:
            print(f"[daily] no bars returned for {TAIEX_SYMBOL}", file=sys.stderr)
        else:
            p = save_daily(df)
            print(f"[daily] +{len(df)} rows  range={df['date'].min()}..{df['date'].max()}  → {p}")


if __name__ == "__main__":
    main()
