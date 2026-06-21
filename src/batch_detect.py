"""
Batch node detection — runs detect_nodes on all crops/ files from
page 14 onwards and writes results to detections_part2.json.

Usage:
    python batch_detect.py
    python batch_detect.py --start-page 14 --out detections_part2.json
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.node_detector import detect_nodes

CROPS_DIR = Path("data/crops/")
DEFAULT_OUT = Path("detections_part2.json")


def page_number(p: Path) -> int:
    """Extract the page index from a filename like page014_cell003.png."""
    return int(p.stem.split("_")[0].replace("page", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch node detection on crops/")
    parser.add_argument("--start-page", type=int, default=14,
                        help="First page index to process (default: 14)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSON path (default: detections_part2.json)")
    args = parser.parse_args()

    crops = sorted(
        p for p in CROPS_DIR.glob("page*.png")
        if page_number(p) >= args.start_page and "_debug" not in p.name
    )

    if not crops:
        print(f"No crops found in {CROPS_DIR}/ from page {args.start_page}+", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(crops)} crops from page {args.start_page}+ ...")

    results: list[dict] = []
    t0 = time.time()
    errors = 0

    for i, crop_path in enumerate(crops, 1):
        try:
            nodes = detect_nodes(str(crop_path))
            results.append({
                "crop": crop_path.name,
                "page": page_number(crop_path),
                "nodes": [asdict(n) for n in nodes],
                "node_count": len(nodes),
            })
        except Exception as exc:
            errors += 1
            results.append({
                "crop": crop_path.name,
                "page": page_number(crop_path),
                "nodes": [],
                "node_count": 0,
                "error": str(exc),
            })

        if i % 100 == 0 or i == len(crops):
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(crops) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(crops)}  ({rate:.1f} crops/s, ~{remaining:.0f}s remaining)")

    args.out.write_text(json.dumps(results, indent=2))

    total_nodes = sum(r["node_count"] for r in results)
    empty = sum(1 for r in results if r["node_count"] == 0)
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  Crops processed : {len(crops)}")
    print(f"  Total nodes     : {total_nodes}")
    print(f"  Empty crops     : {empty}  (no nodes detected)")
    print(f"  Errors          : {errors}")
    print(f"  Output          : {args.out}")


if __name__ == "__main__":
    main()
