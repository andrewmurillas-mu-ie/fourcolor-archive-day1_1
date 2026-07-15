from validator import Validator
import json

# ---------------------------------------------------------------------------
# Hard-coded Configuration #1 — Ring-5 wheel (simplest valid configuration)
#
# One interior vertex (degree 5, "solid dot") surrounded by a ring of 5.
# node_count=1 (interior only), ring_size=5, edge_count=0, e_attachment=10
# Identity check: 3*(1+5) - 5 - 3 = 10  ✓
# ---------------------------------------------------------------------------
CONFIG_1 = {
    "crop": "hardcoded_config_1",
    "page": 0,
    "nodes": [
        {"x": 0, "y": 0, "radius": 10, "shape": "solid_dot", "degree": 5},
    ],
    "node_count": 1,
    "edges": [
        {"from": 0, "to": 1}, {"from": 0, "to": 2}, {"from": 0, "to": 3},
        {"from": 0, "to": 4},
    ],
    "edge_count": 0,
    "ring_size": 5,
    "e_attachment": 10,
    "euler_valid": True,
    "validation_note": "r=5, E_att=10",
}

# ---------------------------------------------------------------------------
# Deliberately broken configuration — one attachment edge missing.
# ---------------------------------------------------------------------------
CONFIG_BAD = {
    "crop": "hardcoded_config_bad",
    "page": 0,
    "nodes": [
        {"x": 0, "y": 0, "radius": 10, "shape": "solid_dot", "degree": 5},
    ],
    "node_count": 1,
    "edges": [],
    "edge_count": 0,
    "ring_size": 5,
    "e_attachment": 9,   # one edge missing → should fail
    "euler_valid": False,
    "validation_note": "r=5, E_att=9 (bad)",
}


def run_validation(cfg: dict, validator: Validator) -> None:
    if cfg.get("ring_size") is None or cfg.get("e_attachment") is None:
        note = cfg.get("validation_note", "ring_size/e_attachment missing")
        print(f"Config {cfg['crop']}")
        print(f"  [SKIP] {note}")
        print()
        return
    V = cfg["node_count"] + cfg["ring_size"]
    result = validator.check(
        V=V,
        E_internal=cfg["edge_count"],
        E_attachment=cfg["e_attachment"],
        r=cfg["ring_size"],
    )
    print(f"Config {cfg['crop']}")
    print(f"  {result}")
    print()


if __name__ == "__main__":
    v = Validator()
    configs = json.load(open("../data/detections_part.json"))
    for cfg in configs:
        run_validation(cfg, v)
    # run_validation(CONFIG_BAD, v)