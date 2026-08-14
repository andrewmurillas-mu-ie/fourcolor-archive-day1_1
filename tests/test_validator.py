"""Fail-fast validator tests (merged from the day-1 archive repo).

Regression anchor: the Birkhoff diamond, which the June batch run passed
INCORRECTLY at page014_cell000 (4/5 edges detected, ring inferred as 7).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.configuration import (
    Configuration, Vertex, birkhoff_diamond, single_five_vertex,
)
from src.validator import Validator, validate, validate_detection


# ------------------------------------------------------------------ #
# Ground truth passes
# ------------------------------------------------------------------ #
def test_birkhoff_diamond_passes():
    rep = validate(birkhoff_diamond())
    assert rep.ok, rep
    assert rep.computed["implied_ring"] == 6
    assert rep.computed["E_total"] == 21
    assert rep.computed["E_total"] == 3 * rep.computed["V_total"] - 6 - 3


def test_single_five_vertex_passes():
    rep = validate(single_five_vertex())
    assert rep.ok, rep
    assert rep.computed["implied_ring"] == 5


def test_legacy_check_api_stable():
    # hitl_ui.py depends on this exact call signature
    res = Validator().check(V=10, E_internal=5, E_attachment=16, r=6)
    assert res.is_valid
    res = Validator().check(V=10, E_internal=5, E_attachment=15, r=6)
    assert not res.is_valid


# ------------------------------------------------------------------ #
# THE regression: June false positive at page014_cell000
# ------------------------------------------------------------------ #
JUNE_DETECTION = {
    # verbatim structure from detections_part.json (edge (2,3) missed by CV)
    "crop": "page014_cell000.png",
    "page": 14,
    "nodes": [
        {"x": 123, "y": 42, "radius": 17, "shape": "solid_dot", "degree": 5},
        {"x": 46, "y": 146, "radius": 19, "shape": "solid_dot", "degree": 5},
        {"x": 192, "y": 146, "radius": 18, "shape": "solid_dot", "degree": 5},
        {"x": 124, "y": 223, "radius": 14, "shape": "solid_dot", "degree": 5},
    ],
    "edges": [{"from": 0, "to": 1}, {"from": 0, "to": 2},
              {"from": 1, "to": 2}, {"from": 1, "to": 3}],
}


def test_june_false_positive_caught_with_labeled_ring():
    rep = validate_detection(JUNE_DETECTION, labeled_ring=6)
    assert not rep.ok
    assert rep.computed["implied_ring"] == 7
    assert rep.computed["ring_delta"] == 1  # exactly one missed edge
    assert any("RING_MISMATCH" in f for f in rep.failures)


def test_june_detection_unverified_without_label():
    # without external ground truth it still structurally passes, but is
    # flagged as unverified — never silently trusted
    rep = validate_detection(JUNE_DETECTION)
    assert rep.ok
    assert any("NO_LABELED_RING" in w for w in rep.warnings)


def test_corrected_detection_passes_with_label():
    det = dict(JUNE_DETECTION)
    det["edges"] = JUNE_DETECTION["edges"] + [{"from": 2, "to": 3}]
    rep = validate_detection(det, labeled_ring=6)
    assert rep.ok, rep
    assert rep.computed["ring_delta"] == 0


# ------------------------------------------------------------------ #
# Structural fail-fast battery
# ------------------------------------------------------------------ #
def test_phantom_edge_detected():
    cfg = birkhoff_diamond()
    cfg.edges = cfg.edges + [(1, 3)]
    rep = validate(cfg)
    assert not rep.ok
    assert rep.computed["ring_delta"] == -1


def test_disconnected_rejected():
    cfg = Configuration(
        id="disconnected",
        vertices=[Vertex(i, 5) for i in range(4)],
        edges=[(0, 1), (2, 3)],
    )
    rep = validate(cfg)
    assert not rep.ok
    assert any("DISCONNECTED" in f for f in rep.failures)


def test_nonplanar_rejected():
    cfg = Configuration(
        id="k5",
        vertices=[Vertex(i, 6) for i in range(5)],
        edges=[(i, j) for i in range(5) for j in range(i + 1, 5)],
    )
    rep = validate(cfg)
    assert not rep.ok
    assert any("NONPLANAR" in f for f in rep.failures)


def test_degree_overflow_rejected():
    cfg = Configuration(
        id="overflow",
        vertices=[Vertex(0, 5)] + [Vertex(i, 5) for i in range(1, 7)],
        edges=[(0, i) for i in range(1, 7)]
        + [(i, i + 1) for i in range(1, 6)] + [(6, 1)],
    )
    rep = validate(cfg)
    assert not rep.ok
    assert any("DEGREE_OVERFLOW" in f for f in rep.failures)


def test_duplicate_edge_and_self_loop_rejected():
    cfg = birkhoff_diamond()
    cfg.edges = cfg.edges + [(1, 0), (2, 2)]
    rep = validate(cfg)
    assert not rep.ok
    assert any("DUPLICATE_EDGE" in f for f in rep.failures)
    assert any("SELF_LOOP" in f for f in rep.failures)


def test_degree_below_five_rejected():
    cfg = birkhoff_diamond()
    cfg.vertices[1] = Vertex(1, 4)
    rep = validate(cfg)
    assert not rep.ok


def test_ring_too_large_fails():
    cfg = Configuration(id="big", vertices=[Vertex(0, 15)], edges=[])
    rep = validate(cfg)
    assert not rep.ok
    assert any("RING_TOO_LARGE" in f for f in rep.failures)


# ------------------------------------------------------------------ #
# Serialization round-trip
# ------------------------------------------------------------------ #
def test_json_round_trip():
    cfg = birkhoff_diamond()
    clone = Configuration.from_json(cfg.to_json())
    assert clone.edges == cfg.edges
    assert validate(clone).ok


def test_shape_degree_consistency_enforced():
    with pytest.raises(ValueError):
        Vertex(0, 6, "solid_dot")
    with pytest.raises(ValueError):
        Vertex(0, 7, "triangle")  # triangle means >= 8
