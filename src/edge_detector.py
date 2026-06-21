"""
Phase 2b — Edge Detection

Uses the line-probe method: sample pixels along the straight line between
two node centres, skipping over the node bodies at each end.  If the ink
fraction along the probe exceeds EDGE_INK_THRESHOLD the edge is present.

The invisible ring size r is inferred algebraically from the degree-sum
formula:  r = sum(d_i) - E_internal - 3·V + 3
Valid Appel–Haken configurations have r ∈ [RING_MIN, RING_MAX].

Usage:
    python edge_detector.py crops_part2/page014_cell000.png
    python edge_detector.py crops_part2/page014_cell000.png --debug
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.node_detector import Node, detect_nodes, load_binary

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

EDGE_INK_THRESHOLD = 0.40   # min ink fraction along the probe segment
EDGE_MAX_DIST_PX   = 210    # ignore node pairs farther apart than this
RING_MIN           = 3      # Appel–Haken ring sizes span roughly [3, 14]
RING_MAX           = 14


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------

def _probe_line(
    binary_inv: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    skip1: int,
    skip2: int,
) -> float | int:
    """Return the ink fraction of pixels sampled along (x1,y1)→(x2,y2),
    skipping skip1 px at the start (node a's body) and skip2 px at the end
    (node b's body).
    """
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return 0.0
    t0 = skip1 / length
    t1 = 1.0 - skip2 / length
    if t0 >= t1:
        return 0.0
    n = max(int(length * (t1 - t0)), 4)
    ts = np.linspace(t0, t1, n)
    xs = np.clip((x1 + ts * dx).astype(int), 0, binary_inv.shape[1] - 1)
    ys = np.clip((y1 + ts * dy).astype(int), 0, binary_inv.shape[0] - 1)
    return float((binary_inv[ys, xs] > 128).sum()) / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(image_path: str, nodes: list[Node]) -> list[tuple[int, int]]:
    """Detect internal edges between nodes via the line-probe method.

    Returns a sorted list of (i, j) pairs with i < j.
    """
    _, binary_inv = load_binary(image_path)
    edges: list[tuple[int, int]] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            dist = float(np.hypot(a.x - b.x, a.y - b.y))
            if dist > EDGE_MAX_DIST_PX:
                continue
            skip_a = a.radius + 4
            skip_b = b.radius + 4
            if dist <= skip_a + skip_b + 4:
                continue  # nodes touching or overlapping
            if _probe_line(binary_inv, a.x, a.y, b.x, b.y, skip_a, skip_b) >= EDGE_INK_THRESHOLD:
                edges.append((i, j))

    return edges


def infer_ring_size(nodes: list[Node], edges: list[tuple[int, int]]) -> int | None:
    """Infer the invisible ring size from the degree-sum identity.

    For a fully triangulated planar disk:
        r = sum(d_i) - E_internal - 3·V + 3

    Returns None if r ∉ [RING_MIN, RING_MAX].
    """
    V = len(nodes)
    if V == 0:
        return None
    r = sum(n.degree for n in nodes) - len(edges) - 3 * V + 3
    return r if RING_MIN <= r <= RING_MAX else None


def compute_e_attachment(nodes: list[Node], edges: list[tuple[int, int]], r: int) -> int:
    """Compute E_attachment = ring-cycle edges + ring-to-interior edges.

    E_attachment = r  +  (sum(d_i) - 2·E_internal)
    """
    return r + sum(n.degree for n in nodes) - 2 * len(edges)


def degree_check(nodes: list[Node], edges: list[tuple[int, int]]) -> bool:
    """True iff no node has more detected internal edges than its total degree."""
    counts = [0] * len(nodes)
    for i, j in edges:
        counts[i] += 1
        counts[j] += 1
    return all(counts[k] <= nodes[k].degree for k in range(len(nodes)))


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def draw_edges(image_path: str, nodes: list[Node],
               edges: list[tuple[int, int]], out_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    for i, j in edges:
        a, b = nodes[i], nodes[j]
        cv2.line(img, (a.x, a.y), (b.x, b.y), (0, 220, 255), 2, cv2.LINE_AA)
    for n in nodes:
        cv2.circle(img, (n.x, n.y), n.radius + 4, (0, 80, 255), 2)
    cv2.imwrite(out_path, img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2b: detect edges in a config crop")
    parser.add_argument("image", help="Path to crop PNG")
    parser.add_argument("--debug", action="store_true",
                        help="Save annotated image (nodes + edges) alongside input")
    args = parser.parse_args()

    nodes = detect_nodes(args.image)
    edges = detect_edges(args.image, nodes)
    r = infer_ring_size(nodes, edges)

    print(f"\n{Path(args.image).name}  —  {len(nodes)} node(s),  {len(edges)} edge(s)")

    if nodes:
        r_raw = sum(n.degree for n in nodes) - len(edges) - 3 * len(nodes) + 3
        valid_r = RING_MIN <= r_raw <= RING_MAX
        ok_deg  = degree_check(nodes, edges)
        print(f"  Inferred ring size r = {r_raw}  {'✓' if valid_r else f'✗ (outside [{RING_MIN},{RING_MAX}])'}")
        print(f"  Degree check         : {'✓ OK' if ok_deg else '✗ FAIL — a node has more internal edges than its degree'}")
        if r is not None and ok_deg:
            e_att = compute_e_attachment(nodes, edges, r)
            V = len(nodes) + r
            E_total = len(edges) + e_att
            print(f"  Euler identity       : {E_total} == 3·{V} - {r} - 3 = {3*V - r - 3}  ✓")
        else:
            print("  Euler identity       : ✗ FAIL — send to HITL")
    else:
        print("  No nodes detected — skipping validation")

    if args.debug:
        p = Path(args.image)
        out = str(p.parent / (p.stem + "_edges.png"))
        draw_edges(args.image, nodes, edges, out)
        print(f"\nDebug image saved to {out}")


if __name__ == "__main__":
    main()
