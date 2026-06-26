"""
Batch pipeline — runs detect_nodes + detect_edges + Euler validation on all
crops from page 14 onwards and writes results to detections_part.json.

Usage:
    python -m src.batch_detect
    python -m src.batch_detect --start-page 14 --out data/detections_part.json
    python -m src.batch_detect --crops data/crops_part2/
"""

import argparse
import json
import sys
import time
import os
from dataclasses import asdict
from pathlib import Path

from src.node_detector import detect_nodes
from src.edge_detector import (
    detect_edges,
    infer_ring_size,
    compute_e_attachment,
    degree_check,
    RING_MIN,
    RING_MAX,
)

from dotenv import load_dotenv

# Load environment  from the .env file
load_dotenv()

def page_number(p: Path) -> int:
    return int(p.stem.split("_")[0].replace("page", ""))


def run_crop(crop_path: Path) -> dict:
    nodes = detect_nodes(str(crop_path))
    edges = detect_edges(str(crop_path), nodes)

    result: dict = {
        "crop": crop_path.name,
        "page": page_number(crop_path),
        "nodes": [asdict(n) for n in nodes],
        "node_count": len(nodes),
        "edges": [{"from": i, "to": j} for i, j in edges],
        "edge_count": len(edges),
        "ring_size": None,
        "e_attachment": None,
        "euler_valid": False,
        "validation_note": "",
    }

    if not nodes:
        result["validation_note"] = "no nodes detected"
        return result

    if not degree_check(nodes, edges):
        r_raw = sum(n.degree for n in nodes) - len(edges) - 3 * len(nodes) + 3
        result["validation_note"] = f"degree violation (r_raw={r_raw})"
        return result

    r = infer_ring_size(nodes, edges)
    if r is None:
        r_raw = sum(n.degree for n in nodes) - len(edges) - 3 * len(nodes) + 3
        result["validation_note"] = f"ring out of range (r_raw={r_raw}, expected [{RING_MIN},{RING_MAX}])"
        return result

    e_att = compute_e_attachment(nodes, edges, r)
    V = len(nodes) + r
    E_total = len(edges) + e_att
    E_expected = 3 * V - r - 3

    if E_total == E_expected:
        result["ring_size"] = r
        result["e_attachment"] = e_att
        result["euler_valid"] = True
        result["validation_note"] = f"r={r}, E_att={e_att}"
    else:
        result["ring_size"] = r
        result["e_attachment"] = e_att
        result["validation_note"] = f"Euler fail: {E_total} != {E_expected} (r={r})"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pipeline: nodes + edges + validation")
    parser.add_argument("--crops", type=Path, default=os.getenv("CROPS_DIR"),
                        help=f"Directory of crop PNGs (default: {os.getenv('CROPS_DIR')})")
    parser.add_argument("--start-page", type=int, default=14,
                        help="First page index to process (default: 14)")
    parser.add_argument("--out", type=Path, default=os.getenv("DEFAULT_OUT"),
                        help=f"Output JSON path (default: {os.getenv('DEFAULT_OUT')})")
    args = parser.parse_args()

    crops = sorted(
        p for p in args.crops.glob("page*.png")
        if page_number(p) >= args.start_page and "_debug" not in p.name
    )

    if not crops:
        print(f"No crops found in {args.crops}/ from page {args.start_page}+", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(crops)} crops from page {args.start_page}+ ...")

    results: list[dict] = []
    t0 = time.time()
    errors = 0

    for i, crop_path in enumerate(crops, 1):
        try:
            results.append(run_crop(crop_path))
        except Exception as exc:
            errors += 1
            results.append({
                "crop": crop_path.name,
                "page": page_number(crop_path),
                "nodes": [],
                "node_count": 0,
                "edges": [],
                "edge_count": 0,
                "ring_size": None,
                "e_attachment": None,
                "euler_valid": False,
                "validation_note": f"error: {exc}",
            })

        if i % 100 == 0 or i == len(crops):
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(crops) - i) / rate if rate > 0 else 0
            valid_so_far = sum(1 for r in results if r.get("euler_valid"))
            print(f"  {i}/{len(crops)}  ({rate:.1f} crops/s, ~{remaining:.0f}s remaining)"
                  f"  valid so far: {valid_so_far}/{i} ({100*valid_so_far/i:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    total = len(results)
    valid = sum(1 for r in results if r.get("euler_valid"))
    no_nodes = sum(1 for r in results if r["node_count"] == 0)
    deg_fail = sum(1 for r in results if "degree violation" in r.get("validation_note", ""))
    ring_fail = sum(1 for r in results if "ring out of range" in r.get("validation_note", ""))
    euler_fail = sum(1 for r in results if "Euler fail" in r.get("validation_note", ""))

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  Crops processed      : {total}")
    print(f"  Auto-valid (PASS)    : {valid} ({100*valid/total:.1f}%)")
    print(f"  No nodes detected    : {no_nodes}")
    print(f"  Degree violations    : {deg_fail}")
    print(f"  Ring out of range    : {ring_fail}")
    print(f"  Euler identity fail  : {euler_fail}")
    print(f"  Errors               : {errors}")
    print(f"  Output               : {args.out}")


if __name__ == "__main__":
    main()