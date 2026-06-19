from validator import Validator

# ---------------------------------------------------------------------------
# Hard-coded Configuration #1 — Ring-5 wheel (simplest valid configuration)
#
# One interior vertex (degree 5, "solid dot" in Appel-Haken notation) surrounded
# by a ring of 5 vertices.
#
#       Ring vertices:  v1, v2, v3, v4, v5  (pentagon)
#       Interior vertex: c  (connected to every ring vertex)
#
#   V            = 6   (5 ring + 1 interior)
#   r            = 5
#   E_attachment = 10  (5 ring-ring edges + 5 ring-interior spokes)
#   E_internal   = 0   (no interior-to-interior edges; only 1 interior vertex)
#
#   Identity check:  3*6 - 5 - 3 = 10  ✓
# ---------------------------------------------------------------------------
CONFIG_1 = {
    "id": 1,
    "label": "Ring-5 single interior vertex (solid dot)",
    "V": 6,
    "r": 5,
    "E_internal": 0,
    "E_attachment": 10,
    "nodes": [
        {"id": 0, "role": "ring"},
        {"id": 1, "role": "ring"},
        {"id": 2, "role": "ring"},
        {"id": 3, "role": "ring"},
        {"id": 4, "role": "ring"},
        {"id": 5, "role": "interior", "shape": "solid_dot", "degree": 5},
    ],
    "edges": [
        # ring cycle
        {"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3},
        {"from": 3, "to": 4}, {"from": 4, "to": 0},
        # interior vertex → ring
        {"from": 5, "to": 0}, {"from": 5, "to": 1}, {"from": 5, "to": 2},
        {"from": 5, "to": 3}, {"from": 5, "to": 4},
    ],
}

# ---------------------------------------------------------------------------
# Deliberately broken configuration — used to confirm the validator rejects
# invalid extractions (simulates a missed edge from the CV pipeline).
# ---------------------------------------------------------------------------
CONFIG_BAD = {
    "id": 99,
    "label": "Ring-5 single interior vertex — one edge missing (bad CV extraction)",
    "V": 6,
    "r": 5,
    "E_internal": 0,
    "E_attachment": 9,   # one edge missing → should fail
}


def run_validation(cfg: dict, validator: Validator) -> None:
    result = validator.check(
        V=cfg["V"],
        E_internal=cfg["E_internal"],
        E_attachment=cfg["E_attachment"],
        r=cfg["r"],
    )
    label = cfg.get("label", f"Config #{cfg['id']}")
    print(f"Config #{cfg['id']}  {label}")
    print(f"  {result}")
    print()


if __name__ == "__main__":
    v = Validator()
    run_validation(CONFIG_1, v)
    run_validation(CONFIG_BAD, v)