from __future__ import annotations

import argparse
import json
from pathlib import Path

from marketpulse.data.providers.ccxt_provider import CCXTCryptoProvider
from marketpulse.data.providers.csv_provider import CSVProvider
from marketpulse.data.providers.yfinance_provider import YahooFinanceProvider
from marketpulse.pipeline import Phase1Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketpulse",
        description="MarketPulse AI Phase 1: historical data pipeline and SMC candidate detection",
    )
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--provider", choices=["auto", "yfinance", "ccxt", "csv"], default="auto")
    parser.add_argument("--symbol", required=True, help="Example: RELIANCE.NS or BTC/USDT")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", default=None, help="End date (exclusive), YYYY-MM-DD; defaults to now")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange id for crypto (default: binance)")
    parser.add_argument("--csv-file", default=None, help="Local OHLCV CSV file when --provider csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--swing-left", type=int, default=3)
    parser.add_argument("--swing-right", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = args.provider
    if provider == "auto":
        provider = "ccxt" if args.asset_class == "crypto" else "yfinance"

    if provider == "yfinance":
        if args.asset_class != "stock":
            raise SystemExit("yfinance provider is intended for --asset-class stock.")
        source_frames = YahooFinanceProvider().fetch(args.symbol, args.start, args.end)
    elif provider == "ccxt":
        if args.asset_class != "crypto":
            raise SystemExit("CCXT provider is intended for --asset-class crypto.")
        source_frames = CCXTCryptoProvider(exchange_id=args.exchange).fetch(
            args.symbol, args.start, args.end
        )
    else:
        if not args.csv_file:
            raise SystemExit("--csv-file is required when --provider csv.")
        source_frames = CSVProvider().fetch(Path(args.csv_file), args.start, args.end)

    manifest = Phase1Pipeline(args.output_dir).run(
        source_frames=source_frames,
        symbol=args.symbol,
        asset_class=args.asset_class,
        swing_left=args.swing_left,
        swing_right=args.swing_right,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
