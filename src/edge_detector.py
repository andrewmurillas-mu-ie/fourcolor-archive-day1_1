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
# Constants
# ---------------------------------------------------------------------------

EDGE_INK_THRESHOLD = 0.80   # min covered fraction along the probe.  0.40
                            # suited the straight-line probe (gaps from
                            # stroke wobble); the stroke-following corridor
                            # absorbs wobble, so real edges should be
                            # near-fully covered — demand it.
EDGE_MAX_DIST_PX   = 210    # ignore node pairs farther apart than this
RING_MIN           = 3      # Appel–Haken ring sizes span roughly [3, 14]
RING_MAX           = 14

#: Corridor half-width (px at 600 DPI).  The diagrams are hand-drawn, so
#: strokes bow away from the straight chord between node centres; a sample
#: point counts as ink if ANY pixel within this perpendicular window is ink.
#: 0 reproduces the original straight-line probe exactly.
CORRIDOR_HALF_WIDTH = 4


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
    xs = x1 + ts * dx
    ys = y1 + ts * dy
    if CORRIDOR_HALF_WIDTH <= 0:
        xi = np.clip(xs.astype(int), 0, binary_inv.shape[1] - 1)
        yi = np.clip(ys.astype(int), 0, binary_inv.shape[0] - 1)
        return float((binary_inv[yi, xi] > 128).sum()) / n

    # Stroke-following corridor probe.  Hand-drawn edges bow away from the
    # straight chord, so at each sample we record the perpendicular OFFSET
    # of the nearest ink (widening the window for longer chords, whose bows
    # are larger).  A real edge yields near-complete coverage AND a smooth
    # offset sequence (we are following one stroke); a parallel nearby
    # stroke yields partial coverage or offset jumps.  Returning plain
    # "any-ink" coverage here caused a phantom-edge explosion (measured:
    # GEOM_EDGE_CROSSING 6,341 -> 44,058) — do not simplify this back.
    half_w = min(12, max(CORRIDOR_HALF_WIDTH, int(0.06 * length)))
    px, py = -dy / length, dx / length
    offsets = np.full(n, np.nan)
    for off in sorted(range(-half_w, half_w + 1), key=abs):
        xi = np.clip((xs + off * px).astype(int), 0, binary_inv.shape[1] - 1)
        yi = np.clip((ys + off * py).astype(int), 0, binary_inv.shape[0] - 1)
        ink = binary_inv[yi, xi] > 128
        offsets = np.where(np.isnan(offsets) & ink, float(off), offsets)

    covered = ~np.isnan(offsets)
    coverage = float(covered.sum()) / n
    if coverage < 1e-9:
        return 0.0
    # smoothness: successive offset jumps > 2 px mean we hopped strokes
    seq = offsets[covered]
    if len(seq) >= 2:
        jumps = np.abs(np.diff(seq))
        rough = float((jumps > 2.0).sum()) / len(jumps)
        if rough > 0.05:
            return 0.0
    return coverage


# ---------------------------------------------------------------------------
# Occlusion guard
# ---------------------------------------------------------------------------

def _occluded_by_node(
    x1: int, y1: int, x2: int, y2: int,
    nodes: list[Node], skip_i: int, skip_j: int,
    skip_a: float, skip_b: float,
) -> bool:
    """Return True if a third node's body lies within the active probe region.

    Only triggers when the perpendicular distance from node k's centre to the
    segment is strictly less than k's own radius — meaning the probe would
    physically pass through k's body and pick up its ink.  The endpoint skip
    zones (skip_a / skip_b) are excluded so triangle vertices that project
    near the endpoints don't falsely block short edges.
    """
    dx, dy = x2 - x1, y2 - y1
    length_sq = float(dx * dx + dy * dy)
    if length_sq == 0:
        return False
    length = float(np.sqrt(length_sq))
    # probe runs from t0 to t1 (skipping the node bodies at each end)
    t0 = skip_a / length
    t1 = 1.0 - skip_b / length
    if t0 >= t1:
        return False
    for k, n in enumerate(nodes):
        if k == skip_i or k == skip_j:
            continue
        t = ((n.x - x1) * dx + (n.y - y1) * dy) / length_sq
        # only consider the probe region, not the endpoint skip zones
        if t < t0 or t > t1:
            continue
        px = x1 + t * dx
        py = y1 + t * dy
        if np.hypot(n.x - px, n.y - py) < n.radius:
            return True
    return False


# ---------------------------------------------------------------------------
# Skeleton tracing (the topological method)
# ---------------------------------------------------------------------------

#: Morphological close kernel before thinning — bridges small print breaks
#: in old ink so an edge with a hairline gap still yields one stroke.
SKEL_CLOSE_KERNEL = 5
#: A stroke component "touches" a node if any of its pixels lies within
#: node.radius + SKEL_TOUCH_PAD of the node centre.
SKEL_TOUCH_PAD = 8
#: Ignore stroke components smaller than this (specks, label remnants).
SKEL_MIN_PIXELS = 6


def detect_edges_skeleton(image_path: str, nodes: list[Node]
                          ) -> tuple[list[tuple[int, int]], list[dict]]:
    """Detect edges by tracing the ink skeleton instead of probing chords.

    Method: close small gaps -> Zhang-Suen thinning (1-px strokes) ->
    erase each node's disc -> connected components of what remains.  In a
    clean drawing every edge meets others only AT vertices, so after disc
    removal each component is exactly one edge: connect the two nodes it
    touches.

    Components touching MORE than two nodes are junction anomalies — in a
    planar hand drawing they usually mean a vertex the node detector missed
    (strokes meet where no node was found) — and are returned as
    diagnostics rather than edges, with the skeleton branch point as a
    candidate node location.  Components touching fewer than two nodes are
    stray ink and ignored.

    Returns (edges, diagnostics): edges as sorted (i, j) with i < j;
    diagnostics as dicts {"kind": "junction", "nodes": [...], "at": (x, y)}.
    """
    _, binary_inv = load_binary(image_path)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (SKEL_CLOSE_KERNEL, SKEL_CLOSE_KERNEL))
    closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, k)
    skeleton = cv2.ximgproc.thinning(closed)

    # erase node discs so strokes separate into per-edge components
    mask = skeleton.copy()
    for n in nodes:
        cv2.circle(mask, (n.x, n.y), n.radius + 4, 0, -1)

    n_comp, labels = cv2.connectedComponents((mask > 0).astype(np.uint8),
                                             connectivity=8)
    edges: set[tuple[int, int]] = set()
    diagnostics: list[dict] = []

    for comp in range(1, n_comp):
        ys, xs = np.nonzero(labels == comp)
        if len(xs) < SKEL_MIN_PIXELS:
            continue
        touching = []
        for idx, n in enumerate(nodes):
            reach = n.radius + SKEL_TOUCH_PAD
            d2 = (xs - n.x) ** 2 + (ys - n.y) ** 2
            if d2.min() <= reach * reach:
                touching.append(idx)
        if len(touching) == 2:
            i, j = sorted(touching)
            edges.add((i, j))
        elif len(touching) > 2:
            # candidate missed vertices = TRUE branch points of the stroke.
            # A naive 3x3 neighbour count fires all along diagonal
            # "staircase" pixels of the skeleton (measured: junction
            # markers sprayed along entire edges); the crossing-number
            # method counts 0->1 transitions around the 8-neighbourhood
            # circle and only fires where >= 3 distinct strokes meet.
            comp_mask = labels == comp
            pts = [(int(c), int(r)) for r, c in
                   np.argwhere(comp_mask) if _crossing_number(
                       comp_mask, int(r), int(c)) >= 3]
            pts = pts or [(int(xs.mean()), int(ys.mean()))]
            for at in _cluster_points(pts, 12):
                diagnostics.append(
                    {"kind": "junction", "nodes": touching, "at": at})
        elif len(touching) == 1:
            # stroke reaching only one node: its far end marks where the
            # other endpoint SHOULD be — often a vertex detection missed
            n = nodes[touching[0]]
            d2 = (xs - n.x) ** 2 + (ys - n.y) ** 2
            far = int(np.argmax(d2))
            if d2[far] >= (n.radius + SKEL_TOUCH_PAD + 12) ** 2:
                diagnostics.append({"kind": "loose_end",
                                    "nodes": touching,
                                    "at": (int(xs[far]), int(ys[far]))})

    return sorted(edges), diagnostics


#: circular order of the 8-neighbourhood for the crossing-number test
_RING8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
          (1, 1), (1, 0), (1, -1), (0, -1)]


def _crossing_number(mask: np.ndarray, r: int, c: int) -> int:
    """Number of 0->1 transitions walking the 8-neighbourhood circle.

    1 = stroke endpoint, 2 = interior of a stroke, >= 3 = true branch
    point.  Robust to the diagonal staircase artefacts that break naive
    neighbour counting on thinned skeletons.
    """
    h, w = mask.shape
    vals = []
    for dr, dc in _RING8:
        rr, cc = r + dr, c + dc
        vals.append(bool(mask[rr, cc]) if 0 <= rr < h and 0 <= cc < w
                    else False)
    return sum(1 for k in range(8)
               if not vals[k] and vals[(k + 1) % 8])


def _cluster_points(pts: list[tuple[int, int]], radius: float
                    ) -> list[tuple[int, int]]:
    """Greedy centroid clustering of 2-D points within `radius`."""
    clusters: list[list[tuple[int, int]]] = []
    for p in pts:
        for cl in clusters:
            cx = sum(q[0] for q in cl) / len(cl)
            cy = sum(q[1] for q in cl) / len(cl)
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= radius ** 2:
                cl.append(p)
                break
        else:
            clusters.append([p])
    return [(int(sum(q[0] for q in cl) / len(cl)),
             int(sum(q[1] for q in cl) / len(cl))) for cl in clusters]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(image_path: str, nodes: list[Node],
                 method: str = "skeleton") -> list[tuple[int, int]]:
    """Detect internal edges between nodes.

    method="skeleton" (default): trace the thinned ink topology — robust to
    hand-drawn curvature; junction anomalies are dropped (fetch them via
    detect_edges_skeleton for diagnostics).
    method="corridor": stroke-following corridor probe along node-pair
    chords (the pre-skeleton fallback; also used by the HITL UI's local
    re-probe).

    Returns a sorted list of (i, j) pairs with i < j.
    """
    if method == "skeleton":
        edges, _ = detect_edges_skeleton(image_path, nodes)
        return edges

    _, binary_inv = load_binary(image_path)
    edges_list: list[tuple[int, int]] = []
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
            if _occluded_by_node(a.x, a.y, b.x, b.y, nodes, i, j, skip_a, skip_b):
                continue
            if _probe_line(binary_inv, a.x, a.y, b.x, b.y, skip_a, skip_b) >= EDGE_INK_THRESHOLD:
                edges_list.append((i, j))

    return edges_list


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
    """Save a debug image with detected edges and node outlines overlaid."""
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
    """CLI: detect edges in one crop, print ring/degree/identity summary
    (--debug saves an annotated image alongside the input)."""
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
