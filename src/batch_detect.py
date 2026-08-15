"""Batch pipeline — runs detect_nodes + detect_edges + fail-fast validation
on all crops from page 14 onwards and writes results to detections_part.json.

Validation honesty note (2026-08-14 merge)
------------------------------------------
The previous version derived the ring size from the detected topology and
then "checked" the Euler identity against that same ring — which holds by
construction, so the old ``Euler fail`` branch was dead code and
``euler_valid`` silently admitted extractions with missed edges (e.g. the
Birkhoff diamond at page014_cell000: 4 of 5 edges found, auto-passed with
ring 7 instead of 6).

``euler_valid`` now means: the full fail-fast battery in src/validator.py
passed (structure, connectivity, planarity, degree consistency, implied ring
within [3, 14]).  The identity itself only becomes an independent test once
a labeled ring size is supplied — see ``--ring-labels`` below.

Usage:
    python -m src.batch_detect
    python -m src.batch_detect --start-page 14 --out data/detections_part.json
    python -m src.batch_detect --crops data/crops/
    python -m src.batch_detect --ring-labels data/ring_labels.json
"""

import argparse
import json
import sys
import time
import os
from dataclasses import asdict
from pathlib import Path

from src.node_detector import detect_nodes
from src.edge_detector import detect_edges
from src.validator import validate_detection, RING_MIN, RING_MAX

from dotenv import load_dotenv

load_dotenv()


def page_number(p: Path) -> int:
    """PDF page index from a crop filename (page014_cell000.png -> 14)."""
    return int(p.stem.split("_")[0].replace("page", ""))


def run_crop(crop_path: Path, labeled_ring: int | None = None) -> dict:
    """Detect nodes+edges in one crop and validate; returns the detection
    dict (nodes, edges, implied ring, failures/warnings, euler_valid)."""
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
        "labeled_ring": labeled_ring,
        "e_attachment": None,
        "euler_valid": False,
        "validation_note": "",
        "failures": [],
        "warnings": [],
    }

    if not nodes:
        result["validation_note"] = "no nodes detected"
        result["failures"] = ["EMPTY: no nodes detected"]
        return result

    rep = validate_detection(result, labeled_ring=labeled_ring)
    result["failures"] = rep.failures
    result["warnings"] = rep.warnings
    result["euler_valid"] = rep.ok
    implied_r = rep.computed.get("implied_ring")
    result["ring_size"] = implied_r
    if implied_r is not None:
        # legacy convention: e_attachment includes the r ring-cycle edges
        result["e_attachment"] = rep.computed["E_att"] + implied_r
    if rep.ok:
        note = f"r={implied_r}, E_att={result['e_attachment']}"
        if labeled_ring is not None:
            note += " (ring verified)"
        result["validation_note"] = note
    else:
        result["validation_note"] = "; ".join(rep.failures)

    return result


def main() -> None:
    """CLI: run the full detect+validate pipeline over a crops directory and
    write detections JSON with an honest per-failure-type summary."""
    parser = argparse.ArgumentParser(
        description="Batch pipeline: nodes + edges + fail-fast validation")
    parser.add_argument("--crops", type=Path, default=os.getenv("CROPS_DIR"),
                        help=f"Directory of crop PNGs (default: {os.getenv('CROPS_DIR')})")
    parser.add_argument("--start-page", type=int, default=14,
                        help="First page index to process (default: 14)")
    parser.add_argument("--out", type=Path, default=os.getenv("DEFAULT_OUT"),
                        help=f"Output JSON path (default: {os.getenv('DEFAULT_OUT')})")
    parser.add_argument("--ring-labels", type=Path, default=None,
                        help="JSON mapping crop filename -> labeled ring size "
                             "(independent ground truth, e.g. from caption "
                             "OCR). Enables the true identity cross-check.")
    args = parser.parse_args()

    ring_labels: dict[str, int] = {}
    if args.ring_labels and args.ring_labels.exists():
        ring_labels = json.loads(args.ring_labels.read_text())
        print(f"Loaded {len(ring_labels)} ring labels from {args.ring_labels}")

    crops = sorted(
        p for p in args.crops.glob("page*.png")
        if page_number(p) >= args.start_page and "_debug" not in p.name
        and "_nodes" not in p.name and "_edges" not in p.name
    )

    if not crops:
        print(f"No crops found in {args.crops}/ from page {args.start_page}+",
              file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(crops)} crops from page {args.start_page}+ ...")

    results: list[dict] = []
    t0 = time.time()
    errors = 0

    for i, crop_path in enumerate(crops, 1):
        try:
            results.append(
                run_crop(crop_path, ring_labels.get(crop_path.name)))
        except Exception as exc:
            errors += 1
            results.append({
                "crop": crop_path.name,
                "page": page_number(crop_path),
                "nodes": [], "node_count": 0,
                "edges": [], "edge_count": 0,
                "ring_size": None, "labeled_ring": None,
                "e_attachment": None,
                "euler_valid": False,
                "validation_note": f"error: {exc}",
                "failures": [f"ERROR: {exc}"],
                "warnings": [],
            })

        if i % 100 == 0 or i == len(crops):
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(crops) - i) / rate if rate > 0 else 0
            valid_so_far = sum(1 for r in results if r.get("euler_valid"))
            print(f"  {i}/{len(crops)}  ({rate:.1f} crops/s, "
                  f"~{remaining:.0f}s remaining)"
                  f"  valid so far: {valid_so_far}/{i} "
                  f"({100 * valid_so_far / i:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    total = len(results)
    valid = sum(1 for r in results if r.get("euler_valid"))
    verified = sum(1 for r in results
                   if r.get("euler_valid") and r.get("labeled_ring") is not None)
    no_nodes = sum(1 for r in results if r["node_count"] == 0)

    def count_failure(tag: str) -> int:
        return sum(1 for r in results
                   if any(f.startswith(tag) for f in r.get("failures", [])))

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  Crops processed          : {total}")
    print(f"  Auto-valid (structural)  : {valid} ({100 * valid / total:.1f}%)")
    print(f"  ...of which ring-verified: {verified} "
          f"(labeled ring available and matching)")
    print(f"  No nodes detected        : {no_nodes}")
    print(f"  Degree overflow          : {count_failure('DEGREE_OVERFLOW')}")
    print(f"  Ring out of range        : "
          f"{count_failure('RING_TOO_SMALL') + count_failure('RING_TOO_LARGE')}")
    print(f"  Ring mismatch (labeled)  : {count_failure('RING_MISMATCH')}")
    print(f"  Disconnected             : {count_failure('DISCONNECTED')}")
    print(f"  Nonplanar                : {count_failure('NONPLANAR')}")
    print(f"  Errors                   : {errors}")
    print(f"  Output                   : {args.out}")


if __name__ == "__main__":
    main()
