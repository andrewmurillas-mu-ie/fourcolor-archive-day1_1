"""Measurement harness for detector changes (Phase 0 of the hardening plan).

Two instruments, used together:

1. BATTERY COMPARISON — pass rates and failure histograms of two batch
   detection runs, with per-crop transitions (fixed / broken lists).  This
   is the coverage metric: are more crops surviving the fail-fast battery?

2. GOLDEN SET — a small collection of crops with complete hand-verified
   ground truth (data/golden.json).  This is the precision metric: it
   catches "valid-but-wrong" extractions that the battery cannot (a passing
   graph may still be a different configuration).  Reports node detection
   precision/recall (position match within tolerance), shape accuracy on
   matched nodes, and edge precision/recall (via the node matching).

Never trust a detector change on one instrument alone: the battery can be
gamed by under-detection, the golden set is tiny.  Improve both or explain.

Usage:
    python -m src.evaluate --new data/detections_v3.json \
                           --baseline data/detections_v2.json
    python -m src.evaluate --new data/detections_v3.json --golden-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

NODE_MATCH_PX = 25   # detected node matches golden node within this radius


# ---------------------------------------------------------------------- #
# Battery comparison
# ---------------------------------------------------------------------- #
def failure_histogram(dets: list[dict]) -> Counter:
    """Count failures by tag (the part before ':') across a detection run."""
    hist: Counter = Counter()
    for d in dets:
        for f in d.get("failures", []):
            hist[f.split(":")[0]] += 1
    return hist


def compare_batteries(new: list[dict], base: list[dict]) -> None:
    new_by = {d["crop"]: d for d in new}
    base_by = {d["crop"]: d for d in base}
    common = sorted(set(new_by) & set(base_by))

    n_pass = sum(1 for c in common if new_by[c].get("euler_valid"))
    b_pass = sum(1 for c in common if base_by[c].get("euler_valid"))
    fixed = [c for c in common
             if new_by[c].get("euler_valid") and not base_by[c].get("euler_valid")]
    broken = [c for c in common
              if not new_by[c].get("euler_valid") and base_by[c].get("euler_valid")]

    print(f"BATTERY  ({len(common)} crops in both runs)")
    print(f"  pass: {b_pass} -> {n_pass}   "
          f"({'+' if n_pass >= b_pass else ''}{n_pass - b_pass})")
    print(f"  fixed: {len(fixed)}   broken: {len(broken)}")
    if broken[:8]:
        print(f"  broken samples: {broken[:8]}")

    nh, bh = failure_histogram(new), failure_histogram(base)
    print("  failure histogram (new vs baseline):")
    for tag in sorted(set(nh) | set(bh), key=lambda t: -bh.get(t, 0)):
        print(f"    {tag:32} {bh.get(tag, 0):5} -> {nh.get(tag, 0):5}")


# ---------------------------------------------------------------------- #
# Golden-set scoring
# ---------------------------------------------------------------------- #
def match_nodes(golden_nodes: list[dict], det_nodes: list[dict]
                ) -> dict[int, int]:
    """Greedy nearest matching golden index -> detected index (within
    NODE_MATCH_PX); each detected node used at most once."""
    pairs = []
    for gi, g in enumerate(golden_nodes):
        for di, d in enumerate(det_nodes):
            dist = ((g["x"] - d["x"]) ** 2 + (g["y"] - d["y"]) ** 2) ** 0.5
            if dist <= NODE_MATCH_PX:
                pairs.append((dist, gi, di))
    pairs.sort()
    used_g, used_d, out = set(), set(), {}
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di); out[gi] = di
    return out


def score_golden(golden: list[dict], dets: list[dict]) -> None:
    det_by = {d["crop"]: d for d in dets}
    tp_n = fp_n = fn_n = shape_ok = shape_tot = 0
    tp_e = fp_e = fn_e = 0
    for g in golden:
        d = det_by.get(g["crop"])
        if d is None:
            print(f"  golden crop missing from run: {g['crop']}")
            continue
        m = match_nodes(g["nodes"], d["nodes"])
        tp_n += len(m)
        fn_n += len(g["nodes"]) - len(m)
        fp_n += len(d["nodes"]) - len(m)
        for gi, di in m.items():
            shape_tot += 1
            if g["nodes"][gi]["shape"] == d["nodes"][di]["shape"]:
                shape_ok += 1
        # edges via the node matching (golden index space -> detected)
        inv = {di: gi for gi, di in m.items()}
        det_edges = set()
        for e in d["edges"]:
            a, b = inv.get(e["from"]), inv.get(e["to"])
            if a is not None and b is not None:
                det_edges.add(frozenset((a, b)))
            else:
                fp_e += 1  # edge touching an unmatched (spurious) node
        gold_edges = {frozenset(e) for e in g["edges"]}
        tp_e += len(det_edges & gold_edges)
        fp_e += len(det_edges - gold_edges)
        fn_e += len(gold_edges - det_edges)

    def pr(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return f"P={p:.2f} R={r:.2f} (tp={tp} fp={fp} fn={fn})"

    print(f"GOLDEN   ({len(golden)} crops)")
    print(f"  nodes : {pr(tp_n, fp_n, fn_n)}")
    print(f"  shapes: {shape_ok}/{shape_tot} correct on matched nodes")
    print(f"  edges : {pr(tp_e, fp_e, fn_e)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector change measurement")
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--golden", type=Path, default=Path("data/golden.json"))
    ap.add_argument("--golden-only", action="store_true")
    args = ap.parse_args()

    new = json.loads(args.new.read_text())
    if not args.golden_only and args.baseline:
        base = json.loads(args.baseline.read_text())
        compare_batteries(new, base)
        print()
    if args.golden.exists():
        score_golden(json.loads(args.golden.read_text()), new)


if __name__ == "__main__":
    main()
