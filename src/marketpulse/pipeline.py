from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marketpulse.data.normalize import normalize_ohlcv
from marketpulse.data.quality import validate_ohlcv
from marketpulse.data.timeframes import build_timeframes
from marketpulse.smc.structure import confirmed_swings, detect_structure_breaks
from marketpulse.smc.zones import (
    detect_fair_value_gaps,
    detect_order_blocks,
    enrich_supply_demand_zones,
    get_active_zones,
)
from marketpulse.smc.confluence import calculate_confluence, ConfluenceResult


class Phase1Pipeline:
    def __init__(self, output_dir: str | Path = "data/processed"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        source_frames: dict[str, pd.DataFrame],
        symbol: str,
        asset_class: str,
        swing_left: int = 3,
        swing_right: int = 3,
    ) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(":", "_").replace(".", "_")

        normalized: dict[str, pd.DataFrame] = {}
        quality: dict[str, dict[str, object]] = {}
        for timeframe, raw in source_frames.items():
            frame = normalize_ohlcv(raw)
            report = validate_ohlcv(frame)
            normalized[timeframe] = frame
            quality[timeframe] = report.to_dict()

        timeframes = build_timeframes(normalized, asset_class=asset_class)
        zone_paths: dict[str, str] = {}
        event_counts: dict[str, dict[str, int]] = {}

        for timeframe, frame in timeframes.items():
            validate_ohlcv(frame)
            frame_path = self.output_dir / f"{safe_symbol}_{timeframe}.csv"
            frame.to_csv(frame_path, index_label="timestamp")

            swings = confirmed_swings(frame, left=swing_left, right=swing_right)
            structure = detect_structure_breaks(frame, swings)
            fvgs = detect_fair_value_gaps(frame)
            order_blocks = detect_order_blocks(frame, structure)
            supply_demand = enrich_supply_demand_zones(frame, swings)

            events_path = self.output_dir / f"{safe_symbol}_{timeframe}_structure.csv"
            zones_path = self.output_dir / f"{safe_symbol}_{timeframe}_zones.csv"
            swings.to_csv(self.output_dir / f"{safe_symbol}_{timeframe}_swings.csv", index=False)
            structure.to_csv(events_path, index=False)

            zone_parts = [part for part in (fvgs, order_blocks, supply_demand) if not part.empty]
            zones = pd.concat(zone_parts, ignore_index=True, sort=False) if zone_parts else pd.DataFrame()
            zones.to_csv(zones_path, index=False)

            # Active (unmitigated + decent strength) zones — useful for live chart later
            active = get_active_zones(zones, min_strength=35.0, only_unmitigated=True)
            active_path = self.output_dir / f"{safe_symbol}_{timeframe}_active_zones.csv"
            active.to_csv(active_path, index=False)

            zone_paths[timeframe] = str(zones_path)
            event_counts[timeframe] = {
                "candles": len(frame),
                "swings": len(swings),
                "structure_events": len(structure),
                "fvgs": len(fvgs),
                "order_blocks": len(order_blocks),
                "supply_demand_zones": len(supply_demand),
                "active_zones": len(active),
                "mitigated_zones": int(zones["is_mitigated"].sum()) if not zones.empty and "is_mitigated" in zones.columns else 0,
            }

        manifest = {
            "project": "MarketPulse AI",
            "phase": 1,
            "symbol": symbol,
            "asset_class": asset_class,
            "timeframes": list(timeframes.keys()),
            "quality_reports": quality,
            "results": event_counts,
            "zone_files": zone_paths,
            "notes": [
                "Timestamps are normalized to UTC.",
                "Current/incomplete candles should be excluded before live decisions.",
                "Swing pivots are emitted only after right-side confirmation candles.",
                "SMC zones are deterministic research candidates, not trade recommendations.",
            ],
        }
        manifest_path = self.output_dir / f"{safe_symbol}_phase1_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        manifest["manifest_file"] = str(manifest_path)
        return manifest

    def run_with_confluence(
        self,
        source_frames: dict[str, pd.DataFrame],
        symbol: str,
        asset_class: str,
        swing_left: int = 3,
        swing_right: int = 3,
    ) -> dict[str, object]:
        """
        Run full Phase 1 pipeline + Confluence analysis.
        Returns both the normal manifest and a clean confluence decision card.
        """
        manifest = self.run(
            source_frames=source_frames,
            symbol=symbol,
            asset_class=asset_class,
            swing_left=swing_left,
            swing_right=swing_right,
        )

        # Rebuild normalized timeframes for confluence
        normalized: dict[str, pd.DataFrame] = {}
        for timeframe, raw in source_frames.items():
            frame = normalize_ohlcv(raw)
            normalized[timeframe] = frame

        timeframes = build_timeframes(normalized, asset_class=asset_class)

        confluence: ConfluenceResult = calculate_confluence(
            symbol=symbol,
            timeframes=timeframes,
            swing_left=swing_left,
            swing_right=swing_right,
        )

        confluence_dict = confluence.to_dict()
        safe_symbol = symbol.replace("/", "_").replace(":", "_").replace(".", "_")
        conf_path = self.output_dir / f"{safe_symbol}_confluence.json"
        conf_path.write_text(json.dumps(confluence_dict, indent=2, default=str), encoding="utf-8")

        manifest["confluence"] = confluence_dict
        manifest["confluence_file"] = str(conf_path)
        return manifest
