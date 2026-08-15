"""Ring-label ground truth extraction from the Part II PDF.

Why this exists
---------------
Table U (Figures 1-63, PDF pages 14-76) prints each configuration with a
position number and a C/D reducibility letter, but NOT its ring size.

IMPORTANT NEGATIVE RESULT (2026-08-14): we first hypothesised the table was
sorted by ring size, which would have let p. 504's ring-size census
(<=8: 7, 9: 8, 10: 35, 11: 89, 12: 334, 13: 701, 14: 660; total 1,834)
assign every configuration a ring label by cumulative rank.  Hand
verification of Figure 5 position 1 refuted this (implied ring ~8 in the
"ring 11" rank zone), and the primary source settles it — journal p. 503:
the table is "organized into primary parts determined by the number of
major vertices of each degree" (Figure 1: no major vertices; Figure 16: a
pair of V7s; Figure 45: two V9s), and within classes the ordering "is more
arbitrary and not wholly consistent".  Ring size is NOT recoverable from
table position.

What this module still provides (all verified):
  * crop -> (figure, position) mapping via the deterministic slot grid
    (pixel-identical replication of the preprocessor's boxes);
  * C/D reducibility letters from the embedded text layer (Gold tier needs
    the D subset);
  * the list of diagrams the preprocessor never cropped (occupancy by ink);
  * total diagram count 1,834 == census total (strong structural check);
  * the census itself as a DATASET-LEVEL acceptance test: the ring-size
    histogram of the final verified dataset must equal it exactly.
Per-configuration ring ground truth must come from HITL confirmation (or
the microfiche supplement, which we do not have); those labels feed
batch_detect --ring-labels as they accumulate.

Method
------
1. Replicate the preprocessor's deterministic box detection (same 600 DPI
   render, deskew, contour filter, reading-order sort), so box k on page p
   IS crop ``page{p:03d}_cell{k:03d}.png`` — verified pixel-identical.
2. Fit the global 5x7 slot grid by clustering box centres across ALL pages
   (the typeset layout is consistent).
3. Per page, decide slot occupancy by INK DENSITY, not by detected boxes —
   the preprocessor missed some diagrams (e.g. Figure 1 position 4), and a
   missed diagram still occupies a table rank.  Empty slots (deleted
   redundancies keep their printed numbers) and "see f-p" cross-references
   contain only label-sized ink and fall under the threshold.
4. Cross-check slot indices against the PDF's embedded text layer (itself
   OCR; digits ~95% reliable) and report disagreements.

Outputs (under --out):
    ring_labels.json   crop filename -> HAND-VERIFIED ring size only
                       (from MANUAL_RING_LABELS; grows via HITL)
    config_index.json  full per-slot records: figure, position, table rank,
                       C/D letter (when readable), crop or MISSING marker

Usage:
    python -m src.ring_labels --pdf part2.pdf --out data/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

from src.preprocessor import (
    DPI, PADDING, pdf_page_to_gray, maybe_deskew, binarise, find_config_boxes,
)

FIRST_FIGURE_PAGE = 14          # PDF page index of Figure 1
LAST_FIGURE_PAGE = 76           # PDF page index of Figure 63
N_COLS, N_ROWS = 5, 7           # "35 possible entries" per figure
SCALE = DPI / 72.0              # PDF points -> 600 DPI pixels

#: Ink threshold for "this slot holds a diagram" (600 DPI pixels).  A
#: two-digit label alone is ~1-2k ink px; the smallest diagrams are >8k.
SLOT_INK_MIN = 5000
LABEL_BAND_PX = 200             # bottom band of a slot that holds the labels

#: p. 504 ring-size census of Table U — a DATASET-LEVEL acceptance test:
#: the ring-size histogram of the fully verified dataset must equal this.
#: (NOT usable per-configuration: the table is not sorted by ring size.)
CENSUS_HISTOGRAM = {"<=8": 7, 9: 8, 10: 35, 11: 89, 12: 334, 13: 701, 14: 660}
CENSUS_TOTAL = 1834

#: Hand-verified ring labels (crop filename -> ring).  Grows via HITL.
MANUAL_RING_LABELS: dict[str, int] = {
    "page014_cell000.png": 6,   # Birkhoff diamond (fig 1 pos 1)
    "page014_cell003.png": 5,   # 5-wheel, hub + 5 rim (fig 1 pos 5):
                                # 6x deg-5, E_int=10 -> r=5.  Visually
                                # verified 2026-08-14 during the audit of
                                # BOUNDARY_ATTACHMENT_DEFICIT (detector had
                                # missed pentagon edge 3-5).
}

#: Cross-reference slots ("see f-p" printed instead of a diagram) whose text
#: the embedded OCR garbled beyond automatic detection.  Visually verified:
#: fig 45 pos 3 reads "See 43-12" (OCR gave "3-I").  These hold no
#: configuration and must not consume a table rank.
KNOWN_XREF_SLOTS = {(45, 3)}

NUM_TOKEN_RE = re.compile(r"^[0-9iIlLoO#]{1,2}$")
LETTER_TOKEN_RE = re.compile(r"^[cCdD]$")
DIGIT_FIXES = str.maketrans({"i": "1", "I": "1", "l": "1", "L": "1",
                             "o": "0", "O": "0"})


def _deskew_matrix(gray: np.ndarray) -> np.ndarray | None:
    """The rotation preprocessor.maybe_deskew applies, as a 2x3 matrix."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return None
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return None
    h, w = gray.shape
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)


def _words_600dpi(page, matrix) -> list[tuple[str, float, float]]:
    out = []
    for x0, y0, x1, y1, token, *_ in page.get_text("words"):
        cx, cy = (x0 + x1) / 2 * SCALE, (y0 + y1) / 2 * SCALE
        if matrix is not None:
            v = matrix @ np.array([cx, cy, 1.0])
            cx, cy = float(v[0]), float(v[1])
        out.append((token, cx, cy))
    return out


def _cluster_1d(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """Simple 1-D k-means for grid centre estimation."""
    lo, hi = float(values.min()), float(values.max())
    centres = np.linspace(lo, hi, n_clusters)
    for _ in range(30):
        assign = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        for k in range(n_clusters):
            sel = values[assign == k]
            if len(sel):
                centres[k] = sel.mean()
    return np.sort(centres)


def load_page(pdf_path: str, page_index: int):
    gray = pdf_page_to_gray(pdf_path, page_index)
    if gray.size == 0:
        return None, None, None, None
    matrix = _deskew_matrix(gray)
    gray = maybe_deskew(gray)
    binary = binarise(gray)
    boxes = find_config_boxes(binary)   # reading order == crop cell order
    return gray, binary, boxes, matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract ring-label ground truth")
    ap.add_argument("--pdf", default="part2.pdf")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    pages = range(FIRST_FIGURE_PAGE, LAST_FIGURE_PAGE + 1)

    # ---- pass 1: collect everything -------------------------------- #
    per_page: dict[int, dict] = {}
    all_cx, all_cy = [], []
    for p in pages:
        gray, binary, boxes, matrix = load_page(args.pdf, p)
        if gray is None:
            continue
        per_page[p] = {"binary": binary, "boxes": boxes,
                       "words": _words_600dpi(doc[p], matrix)}
        for x, y, w, h in boxes:
            all_cx.append(x + w / 2)
            all_cy.append(y + h / 2)
        print(f"pass1 page {p}: {len(boxes)} boxes")

    col_centres = _cluster_1d(np.array(all_cx), N_COLS)
    row_centres = _cluster_1d(np.array(all_cy), N_ROWS)
    col_pitch = float(np.diff(col_centres).mean())
    row_pitch = float(np.diff(row_centres).mean())
    print(f"\nglobal grid: cols {np.round(col_centres)}, "
          f"rows {np.round(row_centres)}")

    def slot_rect(row: int, col: int):
        cx, cy = col_centres[col], row_centres[row]
        return (int(cx - col_pitch / 2), int(cy - row_pitch / 2),
                int(col_pitch), int(row_pitch))

    # ---- pass 2: slot occupancy, crops, labels, ranks --------------- #
    entries: list[dict] = []
    for p in pages:
        if p not in per_page:
            continue
        binary = per_page[p]["binary"]
        boxes = per_page[p]["boxes"]
        words = per_page[p]["words"]
        H, W = binary.shape

        # assign detected boxes (== crops, in order) to slots
        slot_of_box: dict[int, tuple[int, int]] = {}
        for k, (x, y, w, h) in enumerate(boxes):
            cx, cy = x + w / 2, y + h / 2
            col = int(np.argmin(np.abs(col_centres - cx)))
            row = int(np.argmin(np.abs(row_centres - cy)))
            slot_of_box[k] = (row, col)

        box_at_slot = {rc: k for k, rc in slot_of_box.items()}
        if len(box_at_slot) != len(slot_of_box):
            print(f"  WARNING page {p}: two boxes mapped to one slot")

        for row in range(N_ROWS):
            for col in range(N_COLS):
                sx, sy, sw, sh = slot_rect(row, col)
                sx = max(0, sx); sy = max(0, sy)
                region = binary[sy:min(H, sy + sh - LABEL_BAND_PX),
                                sx:min(W, sx + sw)]
                ink = int(region.sum() / 255) if region.size else 0
                k = box_at_slot.get((row, col))

                # "see f-p" cross-reference slots are text, not diagrams
                figure = p - FIRST_FIGURE_PAGE + 1
                position_ = row * N_COLS + col + 1
                is_xref = (figure, position_) in KNOWN_XREF_SLOTS or any(
                    t.lower() == "see"
                    and sx <= cx <= sx + sw and sy <= cy <= sy + sh
                    for t, cx, cy in words)

                # For slots with no detected box, demand ink in the slot
                # CENTRE (inset kills neighbour-diagram bleed at the edges)
                if k is None:
                    inset = 110
                    core = binary[sy + inset:min(H, sy + sh - LABEL_BAND_PX),
                                  sx + inset:min(W, sx + sw - inset)]
                    core_ink = int(core.sum() / 255) if core.size else 0
                    occupied = (not is_xref) and core_ink >= 4000
                else:
                    occupied = True
                if not occupied:
                    continue  # empty slot or cross-reference
                position = row * N_COLS + col + 1

                # OCR cross-check: nearest number token in the label band
                sx2, sy2 = sx, sy + sh - LABEL_BAND_PX - 60
                ocr_pos, letter = None, None
                best_d = 1e9
                for token, cx, cy in words:
                    if not (sy2 <= cy <= sy + sh + 80):
                        continue
                    if not (sx - 80 <= cx <= sx + sw + 80):
                        continue
                    if NUM_TOKEN_RE.match(token):
                        fixed = token.translate(DIGIT_FIXES).replace("#", "")
                        if fixed.isdigit() and 1 <= int(fixed) <= 35:
                            d = abs(cx - sx)  # numbers sit at slot left
                            if d < best_d:
                                best_d, ocr_pos = d, int(fixed)
                    elif LETTER_TOKEN_RE.match(token):
                        letter = token.upper()

                entries.append({
                    "page": p,
                    "figure": p - FIRST_FIGURE_PAGE + 1,
                    "position": position,
                    "crop": (f"page{p:03d}_cell{k:03d}.png"
                             if k is not None else None),
                    "missing_crop": k is None,
                    "slot_ink": ink,
                    "ocr_position": ocr_pos,
                    "ocr_agrees": (ocr_pos == position
                                   if ocr_pos is not None else None),
                    "letter": letter,
                })

    # ---- ranks (table order) + hand-verified ring labels ------------- #
    ring_labels: dict[str, int] = {}
    for rank, rec in enumerate(entries, start=1):
        rec["rank"] = rank          # position in table order (bookkeeping)
        rec["ring"] = MANUAL_RING_LABELS.get(rec["crop"] or "")
        if rec["ring"] is not None:
            ring_labels[rec["crop"]] = rec["ring"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ring_labels.json").write_text(json.dumps(ring_labels,
                                                          indent=2))
    (args.out / "config_index.json").write_text(json.dumps(entries, indent=2))

    # ---- report ------------------------------------------------------ #
    n = len(entries)
    missing = [e for e in entries if e["missing_crop"]]
    ocr_checked = [e for e in entries if e["ocr_agrees"] is not None]
    ocr_bad = [e for e in entries if e["ocr_agrees"] is False]
    letters = sum(1 for e in entries if e["letter"])
    print(f"\nOccupied slots (diagrams) : {n}   "
          f"(census expects {CENSUS_TOTAL})")
    print(f"Hand-verified ring labels : {len(ring_labels)} "
          f"(grow these via HITL; census-by-rank labelling is INVALID)")
    print(f"Crops missing (never cut) : {len(missing)}")
    print(f"OCR position checks       : {len(ocr_checked)} "
          f"({len(ocr_bad)} disagree)")
    print(f"C/D letters readable      : {letters}/{n}")
    if ocr_bad[:10]:
        print("OCR disagreements (first 10):")
        for e in ocr_bad[:10]:
            print(f"  fig {e['figure']} slot {e['position']} "
                  f"ocr={e['ocr_position']} crop={e['crop']}")
    if missing[:10]:
        print("Missing crops (first 10):")
        for e in missing[:10]:
            print(f"  fig {e['figure']} pos {e['position']} "
                  f"(rank {e['rank']}, ring {e['ring']})")


if __name__ == "__main__":
    main()
