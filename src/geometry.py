"""Label-free geometric invariants for extracted configurations.

The drawings give us more than a combinatorial graph: they give an EMBEDDING
(pixel coordinates for every vertex).  A correct extraction of a
configuration must be, geometrically, the interior of a triangulated disk as
drawn.  That yields checks that need no external ring label:

1. NO EDGE CROSSINGS — the source is a planar drawing; two detected edges
   that cross as line segments cannot both be real (phantom-edge detector).
2. NEAR-TRIANGULATION — every *internal* face of the drawn graph must be a
   triangle.  A missed internal edge leaves a quadrilateral (or larger)
   hole.  Localized: we report exactly which face is broken, which is what
   the HITL operator needs.
3. INTERIOR ATTACHMENT — a vertex not on the outer boundary cannot reach
   the ring, so its specified degree must equal its internal degree
   (a_v = d_v − deg_int(v) must be 0).
4. BOUNDARY ATTACHMENT — each visit the outer walk makes to a vertex needs
   at least one attachment to triangulate the outer wedge there.
5. RING WALK CROSS-CHECK — walking the outer face, r_walk = Σa_v − L
   (attachment sum minus outer-walk length) must equal the counting formula
   r = Σd − E_int − 3V + 3.  These agree iff the interior is triangulated,
   so a mismatch is a second, independent computation of the same defect.

HONEST LIMITATION (dissertation-worthy): an error that leaves the graph a
*valid but different* configuration is undetectable by interior invariants
alone.  Example: the June false positive at page014_cell000 (Birkhoff
diamond with boundary edge (2,3) missed) becomes triangle+pendant — a
perfectly coherent ring-7 interior.  Errors on the OUTER boundary can
silently transform one valid configuration into another; only external
information (a hand/HITL-verified ring label, the figure's major-vertex
class, or comparison with the source image) can catch those.  Interior
invariants shrink the undetectable error space; they cannot close it.

All checks assume vertex positions are available (they are, for every
CV-extracted configuration); callers should skip geometry and warn when
positions are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    from src.configuration import Configuration
except ImportError:
    from configuration import Configuration

EPS = 1e-9


@dataclass
class GeometryReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    computed: dict = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Rotation system and face traversal
# --------------------------------------------------------------------- #
def rotation_system(cfg: Configuration) -> dict[int, list[int]]:
    """Neighbours of each vertex sorted counterclockwise by drawn angle.

    (Image coordinates have y pointing down, which flips handedness; the
    face traversal below is convention-consistent either way.)
    """
    pos = {v.id: v.pos for v in cfg.vertices}
    adj: dict[int, list[int]] = {v.id: [] for v in cfg.vertices}
    for u, w in cfg.edges:
        adj[u].append(w)
        adj[w].append(u)
    for v, nbrs in adj.items():
        vx, vy = pos[v]
        nbrs.sort(key=lambda u: math.atan2(pos[u][1] - vy, pos[u][0] - vx))
    return adj


def faces(cfg: Configuration) -> list[list[tuple[int, int]]]:
    """All faces of the drawn embedding as lists of directed darts (u, v).

    Standard combinatorial-map traversal: from dart (u, v), the next dart of
    the same face is (v, w) where w follows u in the rotation order at v.
    Every dart belongs to exactly one face; total darts = 2 * E.
    """
    rot = rotation_system(cfg)
    index_at = {(v, u): i for v, nbrs in rot.items()
                for i, u in enumerate(nbrs)}
    unused = {(u, v) for u, nbrs in rot.items() for v in nbrs}
    out: list[list[tuple[int, int]]] = []
    while unused:
        start = min(unused)
        walk = []
        dart = start
        while True:
            walk.append(dart)
            unused.discard(dart)
            u, v = dart
            nbrs = rot[v]
            i = index_at[(v, u)]
            w = nbrs[(i + 1) % len(nbrs)]
            dart = (v, w)
            if dart == start:
                break
        out.append(walk)
    return out


def _face_area(walk: list[tuple[int, int]], pos: dict) -> float:
    """Signed shoelace area of a face walk (repeated vertices are fine)."""
    area = 0.0
    for u, v in walk:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def outer_face_index(face_list, pos, rot) -> int:
    """Identify the outer face without orientation conventions.

    At the leftmost vertex v*, the angular gap that contains the leftward
    ray (pointing away from the whole drawing) belongs to the outer face.
    With neighbours CCW-sorted by atan2 in (-pi, pi], that gap lies between
    the largest-angle neighbour and the smallest-angle one, so the outer
    face contains the dart (last_neighbour, v*).  Robust to y-axis flips
    (test fixtures use y-up maths coords, detections y-down image coords)
    and to area ties from tree-like spurs, which broke the max-|area| rule.
    """
    vstar = min(pos, key=lambda v: (pos[v][0], pos[v][1]))
    last_nbr = rot[vstar][-1]          # largest atan2 angle at v*
    marker = (last_nbr, vstar)
    for i, walk in enumerate(face_list):
        if marker in walk:
            return i
    raise RuntimeError("outer face marker dart not found")  # unreachable


# --------------------------------------------------------------------- #
# Segment intersection (phantom-edge detector)
# --------------------------------------------------------------------- #
def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Proper crossing of open segments (shared endpoints excluded by caller)."""
    d1 = _orient(*p3, *p4, *p1)
    d2 = _orient(*p3, *p4, *p2)
    d3 = _orient(*p1, *p2, *p3)
    d4 = _orient(*p1, *p2, *p4)
    return (((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS))
            and ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)))


def crossing_edges(cfg: Configuration) -> list[tuple[tuple, tuple]]:
    pos = {v.id: v.pos for v in cfg.vertices}
    crossings = []
    edges = cfg.edges
    for i in range(len(edges)):
        a, b = edges[i]
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue  # shared endpoint
            if _segments_cross(pos[a], pos[b], pos[c], pos[d]):
                crossings.append((edges[i], edges[j]))
    return crossings


# --------------------------------------------------------------------- #
# The geometric battery
# --------------------------------------------------------------------- #
def geometric_report(cfg: Configuration) -> GeometryReport:
    """Run all label-free geometric invariants.  Requires positions."""
    rep = GeometryReport(ok=True)
    fail = rep.failures.append

    if any(v.pos is None for v in cfg.vertices):
        rep.warnings.append("GEOMETRY_UNAVAILABLE: missing vertex positions")
        return rep

    if len(cfg.vertices) == 1 or not cfg.edges:
        rep.computed["faces"] = 0
        return rep  # trivial configurations have no drawn faces to check

    pos = {v.id: v.pos for v in cfg.vertices}

    # 1 — phantom edges cross real ones
    for e1, e2 in crossing_edges(cfg):
        fail(f"GEOM_EDGE_CROSSING: drawn edges {e1} and {e2} intersect — "
             f"at least one is phantom")

    face_list = faces(cfg)
    outer_i = outer_face_index(face_list, pos, rotation_system(cfg))
    outer = face_list[outer_i]
    rep.computed["faces"] = len(face_list)
    rep.computed["outer_walk_len"] = len(outer)

    # 2 — every internal face must be a triangle
    for i, walk in enumerate(face_list):
        if i == outer_i:
            continue
        if len(walk) != 3:
            verts = [u for u, _ in walk]
            fail(f"NONTRIANGULAR_INTERIOR_FACE: face {verts} has "
                 f"{len(walk)} sides — missed internal edge in this hole")

    # 3/4 — attachment accounting
    deg_int = cfg.internal_degree()
    attach = {v.id: v.degree - deg_int[v.id] for v in cfg.vertices}
    boundary_visits: dict[int, int] = {}
    for u, _ in outer:
        boundary_visits[u] = boundary_visits.get(u, 0) + 1

    for v in cfg.vertices:
        visits = boundary_visits.get(v.id, 0)
        if visits == 0 and attach[v.id] != 0:
            fail(f"INTERIOR_VERTEX_ATTACHED: vertex {v.id} is not on the "
                 f"outer boundary but has {attach[v.id]} unaccounted degree "
                 f"— misread shape or missed edge")
        elif visits > 0 and attach[v.id] < visits:
            fail(f"BOUNDARY_ATTACHMENT_DEFICIT: vertex {v.id} needs >= "
                 f"{visits} ring attachments but degree leaves "
                 f"{attach[v.id]}")

    # 5 — independent ring computation via the outer walk
    sum_a = sum(attach.values())
    r_walk = sum_a - len(outer)
    r_count = (cfg.degree_sum() - cfg.n_internal_edges
               - 3 * cfg.n_vertices + 3)
    rep.computed["ring_walk"] = r_walk
    rep.computed["ring_count"] = r_count
    if r_walk != r_count:
        fail(f"RING_WALK_MISMATCH: outer-walk ring {r_walk} != counting "
             f"ring {r_count} — interior is not a triangulated disk")

    rep.ok = not rep.failures
    return rep
