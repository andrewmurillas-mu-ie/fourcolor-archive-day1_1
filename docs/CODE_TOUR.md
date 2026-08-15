# Code Tour — Four Color Theorem Digitisation Pipeline

*A guided walkthrough for humans. For per-function reference, open
`docs/api/index.html` (regenerate with `python -m pdoc src -o docs/api`).
Last updated: day 2 of the August sprint.*

## The one-paragraph version

We turn scanned pages of Appel & Haken's Table U (1,834 configuration
diagrams) into validated graph records. A **configuration** is the interior
of a triangulated disk: vertices whose shapes encode degrees (solid dot=5,
open circle=6, square=7, triangle=8), edges drawn as ink strokes, and an
*invisible* bounding ring whose size is implied by the mathematics. The
pipeline: slice pages into crops → detect nodes and edges → validate
against invariants → correct failures by hand → emit canonical JSON.

## Data flow

```
part2.pdf ──preprocessor──► data/crops/*.png ──node_detector──┐
                                                              ▼
              batch_detect ◄──edge_detector◄── nodes+edges (detection dict)
                   │
                   ▼ validate_detection()
              validator ──uses──► geometry          (fail-fast battery)
                   │
     pass ─────────┼───────── fail
      ▼            ▼
detections_*.json  hitl_ui (human fixes; save gated on identity)
                   │
                   ▼
       annotations/*.json ──Configuration.from_detection──► canonical dataset
```

## Modules, in reading order

### 1. `src/configuration.py` — the data model
Start here. Three dataclasses:
- **`Vertex`** — id, degree, Heesch `shape`, pixel `pos`, marker `radius`.
  Shape↔degree consistency is enforced at construction (a `solid_dot` with
  degree 6 raises).
- **`Provenance`** — where a record came from (`extraction`: manual / cv /
  cv+hitl, `human_corrected`, source page/crop).
- **`Configuration`** — vertices + internal `edges` + optional labeled
  `ring_size` + `reducibility` ("D"/"C") + provenance. JSON round-trip via
  `to_json`/`from_json` (schema v1.1 — bump `SCHEMA_VERSION` on any field
  change). `from_detection()` bridges the pipeline's raw dicts into the
  model; crucially it takes `labeled_ring` from an *external* argument,
  never from the detection itself (that would be circular).

Reference fixtures `birkhoff_diamond()` and `single_five_vertex()` are the
regression anchors — hand-verified ground truth.

### 2. `src/validator.py` — the fail-fast battery
Two layers:
- **`Validator.check(V, E_internal, E_attachment, r)`** — the original
  Triangulated Disk Identity check (E = 3V − r − 3). API-frozen because
  `hitl_ui` calls it. Only meaningful when `r` comes from an independent
  source (a human, a label) — if r was derived from the same topology, the
  identity holds by construction (the June tautology bug).
- **`validate(cfg)` / `validate_detection(det, labeled_ring=None)`** —
  returns a `ValidationReport` (ok, failures[], warnings[], computed{}).
  Checks, in order: structure (self-loops, duplicates, degree ≥ 5),
  connectivity, abstract planarity, degree overflow, implied ring
  `r = Σd − E_int − 3V + 3` in [3, 14], then the geometric battery (below),
  then — if a labeled ring exists — `ring_delta = implied − labeled`
  (+1 per missed edge, −1 per phantom edge; it literally tells the HITL
  operator what to look for).

### 3. `src/geometry.py` — label-free geometric invariants
Uses the *drawn coordinates* (a planar embedding, not just a graph):
- `rotation_system` (neighbours CCW by angle) → `faces` (combinatorial-map
  traversal) → `outer_face_index` (the angular-gap-at-leftmost-vertex
  trick, robust to y-axis flips).
- Checks: **edge crossings** (two straight strokes crossing ⇒ a phantom),
  **near-triangulation** (every internal face a triangle; a hole = missed
  internal edge, reported by face), **attachment accounting** (interior
  vertices must consume all degree internally; each outer-walk visit needs
  a spare attachment), and the **outer-walk ring** `Σ(d−deg_int) − walk_len`
  which must equal the counting ring.
- **Known limitation (pinned by test):** a boundary-edge loss that yields a
  *valid but different* configuration is invisible to interior checks — the
  Birkhoff-diamond false positive. External info (ring label, figure class)
  is the only cure. Gated on connectivity; skipped without positions.

### 4. `src/preprocessor.py` — Phase 1 (pages → crops)
600 DPI render (PyMuPDF) → optional deskew → adaptive threshold →
dilate+contours → filtered boxes → crops `page{p:03d}_cell{k:03d}.png`.
**Determinism contract:** same PDF ⇒ identical boxes in identical order;
`ring_labels.py` depends on this to map crops to Table U slots. 600 DPI is
load-bearing — every pixel constant downstream is calibrated to it.

### 5. `src/node_detector.py` — Phase 2a (vertices)
Three passes with proximity suppression, in priority order: solid dots
(distance-transform local maxima — a dot's centre is far from background,
an edge stroke's midpoint isn't), open circles (Hough on the rim + hollow
centre + rim-ink tests), squares/triangles (contour polygon approximation).
Emits `Node` records; **list order = node id** everywhere downstream.
Known weakness: circles misread as dots (the −1 ring bias) — Phase-2
hardening target.

### 6. `src/edge_detector.py` — Phase 2b (edges)
Two methods. **Skeleton tracing (default)**: close small ink gaps → Zhang–
Suen thinning → erase node discs → each remaining component is one edge
(connect the two nodes it touches). Components touching ≥3 nodes emit
*junction* diagnostics (candidate missed vertices); strokes touching one
node emit *loose ends*. Near-perfect precision (0 crossing/nonplanar
failures over 1,821 crops). **Corridor probe (fallback)**: stroke-following
sampling along node-pair chords, coverage ≥ 0.80 + offset-smoothness gate.
Also home of `infer_ring_size` and `degree_check`.

### 6b. `src/extract.py` — full extraction + `src/evaluate.py` — measurement
`extract_graph` = node detection + skeleton tracing + gap bridging (facing
loose-end pairs become direct edges), plus sensor-fusion vertex *recovery*
(skeleton anomalies confirmed by a weak node detector) and a corridor edge
mode — both only sane inside `batch_detect`'s **validator-guarded attempt
ladder**: plain extraction first, riskier variants only accepted when they
turn a battery failure into a pass (monotone by construction; ladder-passed
crops carry "verify in HITL" notes). `evaluate.py` compares two batch runs
(pass rates, failure histograms, fixed/broken lists) and scores the
hand-verified golden set (`data/golden.json`) for precision — never trust a
detector change on one instrument alone.

### 7. `src/batch_detect.py` — the batch runner
`python -m src.batch_detect --crops data/crops --out data/detections.json
[--ring-labels data/ring_labels.json]`. Runs detect→validate per crop,
writes detection dicts (nodes, edges, implied ring, failures, warnings),
prints an honest per-failure-type summary. `euler_valid` now means "full
battery passed", and `labeled_ring` enables the true identity cross-check.

### 8. `src/hitl_ui.py` — the human-in-the-loop editor
`python -m src.hitl_ui data/crops --detections data/detections_v2.json`.
Matplotlib editor: click a node to cycle its shape, shift-click to delete,
click an edge to remove it, click empty space to add a node, `A` to add an
edge, `+/-` and `]/[` to adjust r and E_attachment, `S` saves (refuses
until the identity passes), `N/P` to browse. **Schema debt:** its saved
annotation JSON is ad-hoc, not the canonical schema — convert via
`Configuration.from_detection`, or unify next time we touch the UI.

### 9. `src/ring_labels.py` — the Table U index
Maps every crop to its (figure, position) slot via a global 5×7 grid fitted
over the deterministic boxes; reads C/D reducibility letters from the PDF's
embedded text layer; finds diagrams the preprocessor never cropped (ink
occupancy per slot). Outputs `data/config_index.json` (all 1,834 slots) and
`data/ring_labels.json` (hand-verified ring labels only — 2 so far).
Contains the documented **negative result**: Table U is ordered by
major-vertex class (p. 503), *not* ring size, so the p. 504 census can't
label configs by position; it survives as the dataset-level acceptance
histogram (≤8:7, 9:8, 10:35, 11:89, 12:334, 13:701, 14:660).

### 10. `src/main.py` — historical Phase-3 harness
Casey's original "validate one hard-coded configuration" milestone.
Superseded by batch_detect + the battery; kept for provenance.

## Invariants that must not drift

- 600 DPI everywhere; node id = detection order; edges stored `(i, j)` with
  `i < j`; the ring is never stored — always implied, then cross-checked.
- `Validator.check()` signature is frozen (hitl_ui).
- Never source a labeled ring from the detection being validated.
- Schema changes bump `SCHEMA_VERSION`.
- Key maths: `E_total = 3V − r − 3`; `r = Σd − E_int − 3V_int + 3`;
  missed edge ⇒ implied r +1, phantom ⇒ −1, circle-as-dot ⇒ −1.

## Current state & scoreboard

20 tests green (`python -m pytest tests/`). Day-2 scoreboard
(`data/detections_day2_pm.json`): **81/1,821** pass the full battery via
the guarded ladder (June baseline 29 — and those included phantom-built
graphs; honest skeleton-only is 50). Phantom failure classes at zero; the
bulk of remaining failures is node-detection recall on disconnected crops
— HITL passes with the junction/loose-end pointers are the intended next
push, alongside figure-class signatures. Hardening plan: `CLAUDE.md` here,
`claude/cv-hardening-plan.md` in the Claude project.
