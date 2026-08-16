"""Fold HITL annotations back into the project's ground truth.

Each file in annotations/ is a canonical Configuration saved by the HITL
editor after passing the full battery, with the operator's ring size as the
label.  This script:

1. re-validates every annotation (never trust files on disk);
2. merges their ring sizes into data/ring_labels.json (operator label wins
   over an existing entry only with --force);
3. builds/updates data/dataset/ — one canonical JSON per configuration,
   enriched with figure/position and the C/D letter from
   data/config_index.json;
4. prints acceptance progress against the p. 504 census histogram
   (the dataset is complete when the ring histogram matches it exactly).

Usage:
    python -m src.collect_annotations
    python -m src.collect_annotations --annotations annotations/ --force
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from src.configuration import Configuration
    from src.validator import validate
    from src.ring_labels import CENSUS_HISTOGRAM, CENSUS_TOTAL
except ImportError:
    from configuration import Configuration
    from validator import validate
    from ring_labels import CENSUS_HISTOGRAM, CENSUS_TOTAL


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge HITL annotations")
    ap.add_argument("--annotations", type=Path, default=Path("annotations"))
    ap.add_argument("--ring-labels", type=Path,
                    default=Path("data/ring_labels.json"))
    ap.add_argument("--index", type=Path,
                    default=Path("data/config_index.json"))
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    ap.add_argument("--force", action="store_true",
                    help="let annotation ring labels overwrite existing ones")
    args = ap.parse_args()

    labels = (json.loads(args.ring_labels.read_text())
              if args.ring_labels.exists() else {})
    index = {}
    if args.index.exists():
        index = {e["crop"]: e for e in json.loads(args.index.read_text())
                 if e.get("crop")}

    args.dataset.mkdir(parents=True, exist_ok=True)
    ok = bad = conflicts = 0
    for f in sorted(args.annotations.glob("*.json")):
        cfg = Configuration.from_json(f.read_text())
        rep = validate(cfg)
        if not rep.ok:
            print(f"  REJECTED {f.name}: {rep.failures[:2]}")
            bad += 1
            continue
        crop = cfg.provenance.crop or f"{cfg.id}.png"
        meta = index.get(crop, {})
        if meta:
            cfg.provenance.figure_ref = (f"Table U {meta['figure']}-"
                                         f"{meta['position']}")
            cfg.reducibility = cfg.reducibility or meta.get("letter")
        old = labels.get(crop)
        if old is not None and old != cfg.ring_size and not args.force:
            print(f"  CONFLICT {crop}: existing label {old} vs "
                  f"annotation {cfg.ring_size} (keep existing; --force to "
                  f"overwrite)")
            conflicts += 1
        else:
            labels[crop] = cfg.ring_size
        (args.dataset / f"{cfg.id}.json").write_text(cfg.to_json())
        ok += 1

    args.ring_labels.write_text(json.dumps(labels, indent=2))

    # census acceptance progress over the verified dataset
    hist: Counter = Counter()
    for f in args.dataset.glob("*.json"):
        cfg = Configuration.from_json(f.read_text())
        r = cfg.ring_size
        hist["<=8" if r is not None and r <= 8 else r] += 1
    print(f"\nAnnotations merged   : {ok} ok, {bad} rejected, "
          f"{conflicts} label conflicts")
    print(f"Ring labels total    : {len(labels)}")
    print(f"Verified dataset     : {sum(hist.values())}/{CENSUS_TOTAL}")
    print("Census progress      : " + ", ".join(
        f"r{k}: {hist.get(k, 0)}/{v}" for k, v in CENSUS_HISTOGRAM.items()))


if __name__ == "__main__":
    main()
