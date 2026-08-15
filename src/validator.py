"""Fail-fast mathematical validator for extracted configurations.

Two layers:

1. `Validator.check(V, E_internal, E_attachment, r)` — the original identity
   check, API-stable for hitl_ui.py and main.py.  IMPORTANT LIMITATION: when
   r and E_attachment are themselves derived from the detected topology (as
   batch_detect does), the identity holds *by construction* and the check is
   tautological.  It only carries information when r comes from an
   independent source (the HITL operator, an OCR'd caption, or the ring-size
   section of Part II the crop belongs to).

2. `validate_detection(det, labeled_ring=...)` / `validate(cfg)` — the full
   fail-fast battery (merged from the day-1 archive repo):
     - structural: self-loops, duplicate edges, dangling ids, degree >= 5
     - graph-theoretic: connectivity, planarity (networkx)
     - degree consistency: internal degree <= specified degree
     - implied ring size r = Σd − E_int − 3·V_int + 3 in [RING_MIN, RING_MAX]
     - labeled-ring cross-check: implied r == labeled r (THE independent
       check; `ring_delta` = implied − labeled guides HITL correction:
       each missed edge gives +1, each phantom edge −1)

Derivation of the identities: see CLAUDE.md ("The core mathematics").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

try:  # support both `from src.validator import ...` and flat imports
    from src.configuration import Configuration
    from src.geometry import geometric_report
except ImportError:
    from configuration import Configuration
    from geometry import geometric_report

RING_MIN = 3
RING_MAX = 14          # largest ring size in the historical set
MIN_DEGREE = 5


# ====================================================================== #
# Layer 1 — original API (kept stable for hitl_ui.py / main.py)
# ====================================================================== #
@dataclass
class ValidationResult:
    is_valid: bool
    V: int
    r: int
    E_total: int
    E_expected: int

    def __str__(self):
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"[{status}] E_total={self.E_total}, "
            f"expected 3*{self.V} - {self.r} - 3 = {self.E_expected}"
        )


class Validator:
    """Checks the Triangulated Disk Identity E_total = 3V − r − 3.

    Meaningful only when r is supplied independently of the detected
    topology (HITL operator, OCR'd caption, or section ground truth).
    """

    def check(self, V: int, E_internal: int, E_attachment: int,
              r: int) -> ValidationResult:
        """V includes the r ring vertices; E_attachment includes ring-cycle
        and ring-to-interior edges."""
        E_total = E_internal + E_attachment
        E_expected = 3 * V - r - 3
        return ValidationResult(
            is_valid=(E_total == E_expected),
            V=V, r=r, E_total=E_total, E_expected=E_expected,
        )


# ====================================================================== #
# Layer 2 — fail-fast battery
# ====================================================================== #
@dataclass
class ValidationReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    computed: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[{status}] " + ", ".join(
            f"{k}={v}" for k, v in self.computed.items())]
        lines += [f"  FAIL: {f}" for f in self.failures]
        lines += [f"  warn: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate(cfg: Configuration) -> ValidationReport:
    """Run the full fail-fast battery on a canonical Configuration."""
    rep = ValidationReport(ok=True)
    fail = rep.failures.append
    warn = rep.warnings.append

    ids = [v.id for v in cfg.vertices]
    id_set = set(ids)

    # --- structural sanity ------------------------------------------- #
    if not cfg.vertices:
        fail("EMPTY: configuration has no vertices")
        rep.ok = False
        return rep
    if len(id_set) != len(ids):
        fail("DUPLICATE_VERTEX_IDS")
    seen = set()
    for u, w in cfg.edges:
        if u == w:
            fail(f"SELF_LOOP: ({u},{w})")
        if u not in id_set or w not in id_set:
            fail(f"DANGLING_EDGE: ({u},{w}) references unknown vertex")
        key = frozenset((u, w))
        if key in seen:
            fail(f"DUPLICATE_EDGE: ({u},{w})")
        seen.add(key)
    for v in cfg.vertices:
        if v.degree < MIN_DEGREE:
            fail(f"DEGREE_TOO_SMALL: vertex {v.id} has degree "
                 f"{v.degree} < {MIN_DEGREE}")
    if rep.failures:
        rep.ok = False
        return rep  # fail fast: arithmetic below assumes a sane graph

    # --- graph-theoretic checks --------------------------------------- #
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(cfg.edges)

    if g.number_of_nodes() > 1 and not nx.is_connected(g):
        fail("DISCONNECTED: interior graph must be connected")

    planar, _ = nx.check_planarity(g)
    if not planar:
        fail("NONPLANAR: interior graph admits no planar embedding")

    deg_int = cfg.internal_degree()
    for v in cfg.vertices:
        if deg_int[v.id] > v.degree:
            fail(f"DEGREE_OVERFLOW: vertex {v.id} has {deg_int[v.id]} "
                 f"internal edges but specified degree {v.degree}")

    # --- the identity -------------------------------------------------- #
    v_int = cfg.n_vertices
    e_int = cfg.n_internal_edges
    sum_d = cfg.degree_sum()
    e_att = sum_d - 2 * e_int            # interior-to-ring edges only
    implied_r = sum_d - e_int - 3 * v_int + 3
    v_tot = v_int + implied_r
    e_tot = e_int + e_att + implied_r    # + ring-cycle edges

    rep.computed.update(
        V_int=v_int, E_int=e_int, sum_degrees=sum_d, E_att=e_att,
        implied_ring=implied_r, V_total=v_tot, E_total=e_tot,
    )
    # Holds by construction — violation would mean a bug here, not bad data.
    assert e_tot == 3 * v_tot - implied_r - 3, "internal identity error"

    if e_att < 0:
        fail(f"NEGATIVE_ATTACHMENT: E_att={e_att}")
    if implied_r < RING_MIN:
        fail(f"RING_TOO_SMALL: implied ring size {implied_r} < {RING_MIN}")
    elif implied_r > RING_MAX:
        fail(f"RING_TOO_LARGE: implied ring {implied_r} > {RING_MAX} "
             f"(historical max) — likely a missed internal edge or misread "
             f"shape")

    # --- label-free geometric invariants (see geometry.py) ------------- #
    # Only meaningful on a connected drawing: face traversal and the outer
    # walk are undefined across components (and isolated vertices have no
    # rotation order at all).  A disconnected extraction has already failed.
    if not any(f.startswith("DISCONNECTED") for f in rep.failures):
        geo = geometric_report(cfg)
        rep.failures.extend(geo.failures)
        rep.warnings.extend(geo.warnings)
        rep.computed.update(geo.computed)

    # --- the independent cross-check ----------------------------------- #
    if cfg.ring_size is not None:
        rep.computed["ring_delta"] = implied_r - cfg.ring_size
        if implied_r != cfg.ring_size:
            fail(f"RING_MISMATCH: labeled ring size {cfg.ring_size} but "
                 f"interior data implies {implied_r} — "
                 f"{'missed' if implied_r > cfg.ring_size else 'phantom'} "
                 f"edge(s) or misread shape(s) likely")
    else:
        warn("NO_LABELED_RING: implied ring size is unverified — "
             "structural checks only")

    rep.ok = not rep.failures
    return rep


def validate_detection(det: dict,
                       labeled_ring: Optional[int] = None
                       ) -> ValidationReport:
    """Fail-fast validation of a pipeline detection dict.

    `labeled_ring` must come from an INDEPENDENT source (caption OCR, the
    Part II ring-size section, or a HITL operator) — never from
    det["ring_size"], which is derived from the same topology being checked.
    """
    cfg = Configuration.from_detection(det, labeled_ring=labeled_ring)
    return validate(cfg)
