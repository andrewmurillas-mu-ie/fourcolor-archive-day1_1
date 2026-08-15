"""Four Color Theorem digitisation pipeline (MSc capstone, Maynooth).

Digitises the 1,834 reducible configurations of Appel & Haken's Table U
(*Every Planar Map is Four Colorable*, Part II) into a validated,
machine-readable JSON archive.

Data flow::

    Part II PDF
        │  preprocessor      600 DPI render → deskew → cell detection
        ▼
    data/crops/page{p}_cell{k}.png          (1,821 crops)
        │  node_detector     dots / circles / squares / triangles
        │  edge_detector     line-probe + occlusion guard
        ▼
    detection dicts ──── batch_detect ────► data/detections_*.json
        │                     │
        │                     ▼
        │            validator (+ geometry)   fail-fast battery
        ▼                     │
    hitl_ui  ◄── failures ────┘
        │  human corrections; identity must pass to save
        ▼
    annotations/*.json  →  configuration.Configuration  →  canonical dataset

Supporting modules: ``configuration`` (canonical data model, schema v1.1),
``ring_labels`` (Table U slot index, C/D letters, hand-verified ring
labels), ``geometry`` (label-free geometric invariants).

Conventions that must not drift: 600 DPI everywhere; node id = detection
order; edges as (i, j) with i < j; the ring is never stored, only implied
and cross-checked.  See CLAUDE.md for working agreements and
docs/CODE_TOUR.md for a guided walkthrough.
"""
