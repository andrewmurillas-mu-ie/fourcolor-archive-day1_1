# CLAUDE.md — Four Color Theorem Digitisation Project

## Project Overview

This is an MSc Software Engineering capstone project at Maynooth University,
supervised by Dr. Kevin Casey (Computer Science Dept, Eolas Building).

The goal is to build a software pipeline that digitises, validates, and
ultimately re-proves the Four Color Theorem using Appel & Haken's original
1976 proof. This is the 50th anniversary of that proof, making it historically
timely. There is currently no publicly available, machine-readable dataset of
the original Appel–Haken configurations — this project will produce one.

The Four Color Theorem states that any planar map can be coloured with at most
four colours such that no two adjacent regions share a colour. The 1976 proof
by Appel & Haken works by:
1. Identifying an **unavoidable set** of graph configurations (every planar map
   must contain at least one of them).
2. Proving every configuration in that set is **reducible** (cannot appear in
   any minimal 5-chromatic map).

The source material is a scanned copy of the paper:
> K. Appel & W. Haken, "Every Planar Map is Four Colorable", 1976.
> Available in this repo as `Every_Planar_Map_is_Four_Colorable.pdf`

---

## Node Symbol Encoding

In the Appel–Haken diagrams, node shapes encode vertex degree in the
planar triangulation:

| Shape       | Degree |
|-------------|--------|
| Solid Dot   | 5      |
| Open Circle | 6      |
| Square      | 7      |
| Triangle    | 8+     |

The bounding **ring** of each configuration is invisible in the diagrams — its
size `r` is mathematically implied by the internal structure.

---

## Key Mathematical Identity

Every valid extracted configuration must satisfy the **Triangulated Disk
Identity**:

```
E_total = 3V - r - 3
```

Where:
- `V` = number of internal vertices
- `r` = ring size (number of boundary vertices)
- `E_total` = total number of edges (internal + attachment edges to ring)

This identity is derived from Euler's formula for planar graphs. It is the
primary "unit test" for the entire pipeline — any extraction that violates it
must be rejected or flagged for manual correction.

---

## Tiered Project Objectives

### Basic (Core MSc requirement)
Build the **Digitisation & Validation Pipeline**:

1. **CV Extractor** — OpenCV pipeline to:
   - Isolate individual configuration subgraphs from scanned pages
   - Classify node shapes (solid dot / open circle / square)
   - Trace pixel paths to map internal edges

2. **Mathematical Validator** — Apply the Triangulated Disk Identity
   (`E = 3V - r - 3`) to every extracted graph. Reject anything that fails.

3. **Human-in-the-Loop (HITL) UI** — A matplotlib or Tkinter canvas that:
   - Displays the original crop in the background
   - Overlays detected nodes (coloured markers) and edges (lines)
   - Lets the user click to toggle an edge or cycle a node shape
     (Dot → Circle → Square) when the validator fails

4. **Output** — A canonical, machine-readable JSON dataset of all
   configurations.

### Gold (Distinction-level)
Build the **D-Reducibility Engine**:

- Generate all valid 4-colourings of boundary rings up to size 14
  (~200,000 cases)
- For each D-labelled configuration, verify that every boundary colouring
  can be extended into the interior without conflicts
- Use bitwise operations and 64-bit integer masks for performance

### Platinum (Publication-level)
**Full Historical Verification (C-Reducibility)**:

- Implement Kempe-chain recolouring algorithms for C-reducible configurations
- Verify all 1,482 configurations from the original Appel–Haken set
- Use multi-core parallel execution
- Deliverable: a report confirming the reducibility of the complete historical
  set on 2026 hardware

---

## Technical Stack

| Layer             | Tools                                         |
|-------------------|-----------------------------------------------|
| Core language     | Python 3.x                                    |
| Computer vision   | OpenCV (`cv2`), NumPy                         |
| Graph logic       | NetworkX, NumPy                               |
| UI (HITL)         | matplotlib or Tkinter                         |
| Validation        | Custom `Validator` class (Euler identity)     |
| Data format       | JSON (canonical configuration schema)         |
| Performance layer | C++, Rust, or Cython (Gold/Platinum only)     |
| Formal modelling  | Alloy (planar graph model, under exploration) |

---

## Coding Pipeline (Phase by Phase)

### Phase 1 — Image Pre-processing
```python
# Goal: isolate clean ink from scanned page
cv2.adaptiveThreshold(...)      # Gaussian window, not global threshold
cv2.medianBlur(ksize=3)         # Remove salt-and-pepper noise
cv2.findContours(...)           # Detect grid cells
cv2.approxPolyDP(...)           # Four-Point Perspective Transform if skewed
```

### Phase 2 — Node Detection
```python
# Primary: Circle Hough Transform
cv2.HoughCircles(...)

# Fallback for non-circular shapes:
cv2.connectedComponents(...)    # Find blobs
# Circularity metric: C = Perimeter^2 / (4 * pi * Area)
# Solid Dot:   high circularity, high pixel density
# Open Circle: high circularity, hollow centre
# Square:      low circularity → cv2.approxPolyDP corner count
```

### Phase 2 — Edge Detection
```python
# Option A: Line-Probe Method
# Draw virtual line between two node centres, sample pixels along it.
# If black pixel ratio > 0.70 → edge exists.

# Option B: Skeletonization (more deterministic)
cv2.ximgproc.thinning(...)      # Zhang-Suen algorithm → 1px-wide lines
```

### Phase 3 — Validator (First Milestone)
```python
class Validator:
    def check(self, V, E_internal, E_attachment, r):
        E_total = E_internal + E_attachment
        assert E_total == 3 * V - r - 3, (
            f"Euler identity violated: {E_total} != {3*V - r - 3}"
        )
```
**Start here before building the full CV pipeline.** Hard-code one known
configuration, pass it through the Validator, confirm it passes. This gives
you a regression test for every future extraction.

### Phase 4 — HITL Correction UI
```python
# matplotlib or Tkinter canvas:
# - Background: original crop image
# - Overlay: detected nodes as coloured circles, edges as lines
# - Click interaction: toggle edge on/off, cycle node shape
# - Save corrected graph to JSON when Validator passes
```

---

## JSON Output Schema (Canonical Configuration Format)

```json
{
  "id": 1,
  "ring_size": 5,
  "nodes": [
    {"id": 0, "shape": "solid_dot", "degree": 5, "x": 120, "y": 85},
    {"id": 1, "shape": "open_circle", "degree": 6, "x": 200, "y": 85}
  ],
  "edges": [
    {"from": 0, "to": 1},
    {"from": 1, "to": 2}
  ],
  "euler_valid": true,
  "source_page": 476,
  "source_label": "CTL #101"
}
```

---

## Current Progress (as of 14 Aug 2026 — 9-day sprint to submission)

- **Basic tier ~70% built** (June work): src/ pipeline (preprocessor →
  node_detector → edge_detector → batch_detect), full HITL editor
  (hitl_ui.py), 1,804 crops from the Part II PDF (pages 14–76),
  1,821 detections of which 666 auto-pass structural validation.
- **2026-08-14 merge** (from the day-1 fourcolor-archive repo, now retired):
  canonical data model (src/configuration.py, schema v1.1), fail-fast
  validator battery (src/validator.py), tests/ (15 passing),
  requirements.txt. `Validator.check(V, E_internal, E_attachment, r)` is
  API-stable for hitl_ui.py.
- **Validation honesty fix**: the old `euler_valid` was tautological — ring
  size was derived from the detected topology, then the identity "checked"
  against that same ring (dead `Euler fail` branch). Proof it mattered:
  page014_cell000 is the Birkhoff diamond; CV missed edge (2,3) and it
  auto-passed with r=7 (true r=6). `euler_valid` now = full structural
  battery (connectivity, planarity, degree consistency, ring in [3,14]);
  the identity becomes a real test only with an INDEPENDENT labeled ring
  (`batch_detect --ring-labels`, HITL, or caption OCR). `ring_delta` =
  implied − labeled: +1 per missed edge, −1 per phantom edge.
- **Table U index + negative result** (src/ring_labels.py,
  data/config_index.json): all 1,834 configurations mapped to
  figure/position and crops; C/D letters for 1,426 (D: 1,030 — the Gold
  worklist); 13 diagrams never cropped. Census-by-rank ring labelling was
  REFUTED (p. 503: table ordered by major-vertex class, not ring size);
  p. 504 census kept as dataset-level acceptance histogram. Ring labels
  are hand/HITL-verified only (data/ring_labels.json; 2 so far).
- **Label-free geometric invariants** (src/geometry.py, in the validator):
  edge-crossing (phantom edges), near-triangulation of internal faces
  (missed-edge holes, localized per face), interior/boundary attachment
  accounting, outer-walk ring cross-check. Gated on connectivity.
  Honest limitation pinned by test: a boundary-edge loss that yields a
  *valid but different* configuration (the diamond case) is undetectable
  without external info. Honest batch stats (data/detections_v2.json):
  only 29/1,821 raw June detections survive the full battery — June's
  "666 valid" included 240 disconnected graphs and 44 with geometrically
  crossing edges. Detector improvement is the real remaining work:
  disconnection (missed edges) 1,069 crops; then shape accuracy.
- **Day 2 edge detection** (src/extract.py, edge_detector skeleton
  tracer): chord probes replaced by ink-topology tracing (close gaps ->
  Zhang-Suen thinning -> erase node discs -> components = edges). Battery
  pass 29 -> 50; phantom classes eliminated (crossings 6,341 -> 0,
  nonplanar 305 -> 0, degree overflow 681 -> 6). DISCONNECTED rose
  1,069 -> 1,630 honestly: June's connectivity was partly phantom-built.
  Remaining bottleneck = node detection recall (junction/loose-end
  diagnostics point HITL at candidate missed vertices). Junction node
  recovery measured NET-NEGATIVE in all tried gatings -> off by default
  (see src/extract.py docstring). Measurement: src/evaluate.py (battery
  diff + golden set, data/golden.json — grow it via HITL).
- **Day 2 afternoon — validator-guarded attempt ladder** (batch_detect +
  src/extract.py): extract plain first; on battery failure retry with
  sensor-fusion vertex recovery (crossing-number branch points / loose-end
  clusters confirmed by a weak node detector) and/or corridor edges; keep
  a variant only if the battery then passes. Monotone by construction:
  29 (June honest) -> 50 (skeleton) -> **81 passes, 0 broken**. Rungs
  fired: corridor 16, recovery 14, both 1 — all flagged "verify in HITL"
  (battery-pass is not ground truth). batch_detect gained --workers.
- Sibling repo `four-colour-dafny/` (Gonthier-style Dafny hypermaps) is
  FROZEN — dissertation background chapter only.
- Sprint plan: days 1–2 validator/merge (done) + ring-label ground truth
  (OCR captions / section ranges); days 2–4 detector improvements + HITL
  passes on the 1,155 unparsed crops; days 5–6 dataset production + Gold
  D-reducibility engine; days 7–9 dissertation + verification.

## Working agreements (two Claudes, one repo)

- This repo is canonical. Cowork-Claude (cloud) and Claude Code (VS Code)
  both work here; git arbitrates — commit early, small, descriptive.
- Do not change `Validator.check()`'s signature (hitl_ui.py depends on it).
- Schema changes require bumping SCHEMA_VERSION in src/configuration.py.
- Never source a labeled ring from det["ring_size"] — that's circular.
- Keep notable AI contributions logged (Casey's request): commit messages
  or docs/ai-log.md.

---

## Suggested Next Steps

1. **Implement the Validator class** and hard-code Configuration #1 as a test
2. **Write Phase 1 pre-processing** on a single page of the PDF
3. **Write Phase 2 node detection** for solid dots first (simplest case)
4. **Connect Validator to Phase 2 output** and confirm the identity holds
5. **Build the HITL UI** for manual correction
6. **Batch process** all pages to produce the JSON dataset

---

## Reference Material in This Repo

- `Every_Planar_Map_is_Four_Colorable.pdf` — The original Appel & Haken paper
- `MSc_Software_Engineering_Project_The_50th_Anniversary_Archival_Proof.md` —
  Full project specification with grading rubric
- `clean_emails.txt` — Email correspondence with Dr. Casey containing
  detailed phase-by-phase coding guidance

---

## Key Contacts

| Person          | Role       | Email                          |
|-----------------|------------|--------------------------------|
| Dr. Kevin Casey | Supervisor | Kevin.Casey@mu.ie              |
| Andrés Murillas | Student    | ANDRES.MURILLAS.2025@mumail.ie |
