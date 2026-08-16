# HITL Context — Four Color Theorem Digitisation (CS641 Capstone)

*Self-contained briefing on the Human-in-the-Loop workflow. Written for a
new chat/Cowork session (or a person) with access to the repo at
`~/Desktop/fourcolor-archive-day1_1` but no prior conversation history.
State as of day 3 of the August 2026 sprint.*

## Why HITL exists (understanding)

The pipeline digitises the 1,834 configuration diagrams of Appel & Haken's
Table U (*Every Planar Map is Four Colorable*, Part II, Figures 1–63; PDF
pages 14–76) into validated graph records. A **configuration** is the
interior of a triangulated disk: vertices whose Heesch shapes encode
degrees (solid dot = 5, open circle = 6, square = 7, triangle = 8), edges
drawn as hand-drawn ink strokes, and an *invisible* bounding ring whose
size r is implied by the mathematics:

    r = Σd − E_int − 3·V_int + 3        (implied ring size)
    E_total = 3V − r − 3                (Triangulated Disk Identity)

Automated extraction (skeleton tracing + a validator-guarded attempt
ladder) currently auto-passes **81 of 1,821 crops** with essentially
perfect precision (zero phantom edges corpus-wide). The remaining ~1,740
crops fail mostly because the node detector missed vertices — the strokes
are traced correctly but lead to undetected endpoints. Algorithmic
recovery of those vertices measured net-negative in every unguarded
variant tried, so **the human eye is the designated instrument for the
remainder**. That is what HITL is: not a fallback, the plan.

Three facts make HITL time efficient here:

1. **The validator tells you what's wrong.** `ring_delta = implied −
   labeled ring: +1 per missed edge, −1 per phantom edge, −1 per open
   circle misread as a solid dot. A crop failing with delta +2 means "find
   two missing edges."
2. **The pipeline tells you where to look.** Failing crops carry junction
   pointers (strokes meet where no node was detected) drawn as green X
   marks in the editor — candidate missed vertices with coordinates.
3. **Every save does triple duty.** A saved annotation is simultaneously a
   canonical dataset entry, a hand-verified ring label (feeding future
   batch runs' identity cross-check), and golden-set material.

Trust model: the editor's save button is gated on the **full fail-fast
battery** (structure, connectivity, planarity, degree consistency,
geometric near-triangulation, attachment accounting, outer-walk ring
cross-check, and implied-ring == operator's ring). You cannot save an
internally inconsistent graph. But a passing battery is still not ground
truth — a wrong-but-coherent graph can pass — so the human's job is to
match the INK, not to satisfy the panel. Crops auto-passed by the riskier
ladder rungs carry "verify in HITL" notes and appear FIRST in the queue as
fast yes/no confirmations.

## Usage

Setup (once per machine): `pip install -r requirements.txt`
(needs opencv-contrib for the skeleton tracer), then verify with
`python -m pytest tests/` → 20 passed.

Start a triage session from the repo root:

    python -m src.hitl_ui data/crops --detections data/detections_day2_pm.json --queue

- `--queue` orders crops: ladder-passed "verify in HITL" first (quick
  confirms), then failing crops by ascending failure count, wrecks last;
  clean auto-passes are skipped.
- Batch nodes AND edges are preloaded; ring size is preset when a verified
  label exists (`data/ring_labels.json`).
- Green X = junction pointer ("strokes meet here; probably a missed
  vertex").

Controls: click a node → cycle shape (dot→circle→square→triangle);
shift-click → delete node; click empty space → add a solid-dot node; click
an edge → delete it; `A` then two nodes → add an edge (Esc cancels);
`+`/`-` → ring size; `S` → save (refuses until the battery passes, and
prints the failures — read them, they name the defect); `N`/`P` → next /
previous crop; closing the window ends the session.

The info panel runs THE SAME full battery as the save gate (single source
of truth) and shows `ring_delta = implied − yours` live — panel green
means save will succeed. (History note for anyone reading old code: an
earlier revision had the panel on a legacy identity check with a manual
E_attachment counter (`]`/`[` keys) while save ran the battery — they
could disagree. Found in external review 2026-08-16 and unified; the
counter is gone.)

Operator method that works: first make the overlay match the ink (add the
missed vertex at the green X, connect its edges, fix any shape the panel's
degree accounting complains about), then set r until the panel goes green
(panel green now guarantees the save succeeds — same check). If the panel
refuses green with the ink fully matched, the shapes are the suspects — check dots that should be circles (this is the pipeline's most
common error). Never bend the graph away from the ink to satisfy the
panel.

After a session:

    python -m src.collect_annotations

Re-validates every annotation, merges ring labels (existing labels win on
conflict unless `--force`), enriches entries with Table U figure/position
and C/D reducibility from `data/config_index.json`, writes the canonical
dataset to `data/dataset/`, and prints census acceptance progress. The
dataset is COMPLETE when the ring histogram equals p. 504's census exactly:
ring ≤8: 7, 9: 8, 10: 35, 11: 89, 12: 334, 13: 701, 14: 660 (Σ = 1,834).
Commit `annotations/`, `data/dataset/`, and `data/ring_labels.json` after
each session (they are gitignore-excepted deliberately).

## Files and conventions (do not drift)

- Detections: `data/detections_day2_pm.json` (current best batch run).
- Annotations: `annotations/<crop-stem>.json` — canonical schema v1.1
  (`src/configuration.py`), `provenance.extraction = "cv+hitl"`,
  `human_corrected = true`, `ring_size` = operator's verified r.
- Node id = list index; edges stored `(i, j)` with `i < j`; the ring is
  never stored, only implied and cross-checked; 600 DPI everywhere.
- Known editor limitation: no undo — `N`+`P` to reload a crop from the
  batch detection resets your edits.
- Deeper background: `docs/CODE_TOUR.md` (module walkthrough),
  `CLAUDE.md` (working agreements + sprint history), `docs/api/index.html`
  (function reference).