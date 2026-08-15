"""Full-graph extraction with junction-guided node recovery.

The skeleton tracer (edge_detector.detect_edges_skeleton) has near-perfect
edge precision but exposes the pipeline's real weakness: vertices the node
detector missed.  Where strokes meet at an undetected vertex, the skeleton
component touches 3+ known nodes and is reported as a *junction anomaly*
with the branch-point coordinates — which is exactly a candidate node.

This module closes the loop:

    nodes = detect_nodes(crop)
    repeat (<= MAX_RECOVERY_ROUNDS):
        edges, junctions = detect_edges_skeleton(crop, nodes)
        classify the ink at each junction branch point -> new Node
        if no new nodes: stop

Junction classification uses the drawing's own conventions: a large filled
blob is a solid dot (degree 5); a hollow ring is an open circle (degree 6);
a bare stroke junction with no marker is treated as an UNMARKED vertex —
per Part I's Figure 1 legend ("degree 5: no special marking at the
vertex"), that is also degree 5.  Recovered nodes carry ``recovered=True``
so the HITL queue can prioritise eyeballing them.

MEASURED STATUS (2026-08-15 pm): recovery is SENSOR-FUSION v2 — a skeleton
anomaly (true branch point by crossing-number, or a >= 2 loose-end cluster)
only becomes a vertex when the weak node detector independently sees a node
there.  Applied unconditionally this still measured net-negative (50 -> 43
battery passes: weak candidates land on thick strokes and subdivide edges),
so callers must use it VALIDATOR-GUARDED, as batch_detect.run_crop does:
extract without recovery first; on battery failure retry with recovery
(and/or corridor edges) and keep a variant only if the battery then passes.
Guarded attempt ladder measured +31 passes, 0 broken (50 -> 81; June
baseline 29).  Ladder-passed crops carry "verify in HITL" notes: a passing
battery is still not ground truth.  Earlier failures kept for the record:
blind skeleton-evidence recovery 50 -> 26 (81% junk nodes); loose-end
vertices subdivide print gaps (NONTRIANGULAR 58 -> 306) -> gaps BRIDGE
instead.

Usage:
    from src.extract import extract_graph
    nodes, edges, leftovers = extract_graph("data/crops/page014_cell003.png")
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from src.node_detector import (
        Node, detect_nodes, load_binary, _binarise, _fill_ratio,
        _mean_intensity, _ring_ink_density, SOLID_BINARISE_BLOCK,
        PROXIMITY_PX, detect_weak_candidates,
    )
    from src.edge_detector import detect_edges_skeleton, detect_edges
except ImportError:
    from node_detector import (
        Node, detect_nodes, load_binary, _binarise, _fill_ratio,
        _mean_intensity, _ring_ink_density, SOLID_BINARISE_BLOCK,
        PROXIMITY_PX, detect_weak_candidates,
    )
    from edge_detector import detect_edges_skeleton, detect_edges

MAX_RECOVERY_ROUNDS = 3
MIN_RECOVERED_RADIUS = 6      # unmarked junction vertices get this radius


def _classify_at(gray: np.ndarray, binary_solid: np.ndarray,
                 binary: np.ndarray, x: int, y: int) -> Node:
    """Classify the ink at a junction point into a Node.

    Order: solid blob (distance-transform radius + fill) -> hollow ring
    (bright centre + inked rim) -> bare junction (unmarked = degree 5).
    """
    dist = cv2.distanceTransform((binary_solid > 0).astype(np.uint8),
                                 cv2.DIST_L2, 5)
    r_est = int(dist[min(y, dist.shape[0] - 1), min(x, dist.shape[1] - 1)])

    if r_est >= 8 and _fill_ratio(binary_solid, x, y, r_est, 0.6) >= 0.6:
        return Node(x=x, y=y, radius=max(r_est, MIN_RECOVERED_RADIUS),
                    shape="solid_dot", degree=5, recovered=True)

    # hollow ring? probe a plausible open-circle radius
    for r in (14, 17, 20):
        if (_mean_intensity(gray, x, y, r) > 150
                and _ring_ink_density(binary, x, y, r) > 0.5):
            return Node(x=x, y=y, radius=r, shape="open_circle", degree=6,
                        recovered=True)

    # bare stroke junction: unmarked vertex = degree 5 (Part I, Fig. 1)
    return Node(x=x, y=y, radius=MIN_RECOVERED_RADIUS,
                shape="solid_dot", degree=5, recovered=True)


def extract_graph(image_path: str, recover: bool = True,
                  bridge: bool = True, edge_method: str = "skeleton"
                  ) -> tuple[list[Node], list[tuple[int, int]], list[dict]]:
    """Detect nodes, then iterate skeleton tracing + junction node recovery.

    Returns (nodes, edges, leftover_diagnostics).  Recovered nodes are
    appended after detected ones (ids stay stable for the originals) and
    have ``recovered=True``.  Leftover diagnostics are junctions that still
    would not resolve after MAX_RECOVERY_ROUNDS (rare; HITL material).
    """
    gray, binary = load_binary(image_path)
    binary_solid = _binarise(gray, SOLID_BINARISE_BLOCK)

    nodes = detect_nodes(image_path)
    edges: list[tuple[int, int]] = []
    junctions: list[dict] = []

    for _ in range(MAX_RECOVERY_ROUNDS):
        edges, junctions = detect_edges_skeleton(image_path, nodes)
        # node candidates come from junction branch clusters ONLY.  Loose
        # ends are handled after the loop as BRIDGES: a facing pair of
        # loose ends is a print gap in the middle of one edge, and turning
        # it into a vertex subdivides the edge and wrecks the triangulation
        # (measured: NONTRIANGULAR_INTERIOR_FACE 58 -> 306).
        # SENSOR FUSION (v2 recovery): a skeleton anomaly (junction branch
        # point or loose-end cluster) is only accepted as a vertex when the
        # WEAK node detector independently sees a node there; the node is
        # placed at the weak candidate's position (visual localisation
        # beats branch-pixel localisation).  v1 recovery (skeleton evidence
        # alone) measured net-negative — see the docstring history.
        evidence = [j["at"] for j in junctions if j["kind"] == "junction"]
        loose_pts = [j["at"] for j in junctions if j["kind"] == "loose_end"]
        from src.edge_detector import _cluster_points
        evidence += [cl for cl in _cluster_points(loose_pts, 16)
                     if sum(1 for pt in loose_pts
                            if (pt[0] - cl[0]) ** 2 + (pt[1] - cl[1]) ** 2
                            <= 16 ** 2) >= 2]
        new_nodes = []
        if recover and evidence:
            weak = detect_weak_candidates(image_path, nodes)
            for w in weak:
                if any((w.x - ex) ** 2 + (w.y - ey) ** 2 <= 22 ** 2
                       for ex, ey in evidence) and                    not any((w.x - n.x) ** 2 + (w.y - n.y) ** 2
                           < PROXIMITY_PX ** 2 for n in nodes + new_nodes):
                    new_nodes.append(w)
        if not new_nodes:
            break
        nodes = nodes + new_nodes

    # PRUNE-AND-RETRACE.  A genuine recovered vertex must earn >= 2 incident
    # edges; recovered nodes that end isolated or pendant are artefacts
    # (measured before this pruning existed: 3,101 recovered nodes, of
    # which 1,370 isolated + 1,142 pendant = 81% junk, and the battery pass
    # rate DROPPED 50 -> 26).  Pruning changes the components, so re-trace
    # until stable.
    for _ in range(MAX_RECOVERY_ROUNDS):
        deg: dict[int, int] = {}
        for i, j in edges:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        keep = [n for idx, n in enumerate(nodes)
                if not n.recovered or deg.get(idx, 0) >= 2]
        if len(keep) == len(nodes):
            break
        nodes = keep
        edges, junctions = detect_edges_skeleton(image_path, nodes)

    # GAP BRIDGING.  Two loose ends facing each other within a small radius
    # are the frayed halves of one edge whose ink the close kernel could
    # not join: connect their far nodes directly instead of inventing a
    # vertex between them.
    loose = [j for j in junctions if j["kind"] == "loose_end"] if bridge \
        else []
    edge_set = set(edges)
    for a_i in range(len(loose)):
        for b_i in range(a_i + 1, len(loose)):
            a, b = loose[a_i], loose[b_i]
            (ax, ay), (bx, by) = a["at"], b["at"]
            if (ax - bx) ** 2 + (ay - by) ** 2 > 16 ** 2:
                continue
            na, nb = a["nodes"][0], b["nodes"][0]
            if na == nb:
                continue
            edge_set.add(tuple(sorted((na, nb))))
    edges = sorted(edge_set)
    junctions = [j for j in junctions if j["kind"] != "loose_end"]

    if edge_method == "corridor":
        # corridor probing over the (possibly recovery-augmented) node set;
        # only sane inside validator-guarded attempts — unguarded corridor
        # measured heavy phantom rates
        edges = detect_edges(image_path, nodes, method="corridor")

    return nodes, edges, junctions
