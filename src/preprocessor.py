"""
Phase 1 — Image Pre-processing

Converts a single page of the Appel-Haken PDF into clean, cropped sub-images,
one per configuration diagram.

Usage:
    python preprocessor.py                    # processes page 0 of the default PDF
    python preprocessor.py --page 5           # page index (0-based)
    python preprocessor.py --pdf path/to.pdf  # custom PDF path
    python preprocessor.py --out crops/       # output directory for crops

Requires:  pip install pymupdf
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PDF = "../Every planar Map is Four Colorable part2 reducibility.pdf"
DPI = 600          # 600 DPI doubles all pixel distances vs 300 DPI, giving a clear
                   # morphological gap between thin edges (~10px) and solid dots (~20px)
MIN_CELL_AREA = 8_000   # px² — discard tiny noise contours
MAX_CELL_AREA = 1_000_000  # px² — discard full-page contour
ASPECT_RATIO_BOUNDS = (0.4, 2.5)  # width/height range for a plausible config cell
PADDING = 10       # px to add around each crop


# ---------------------------------------------------------------------------
# Step 1 — PDF page → grayscale NumPy array
# ---------------------------------------------------------------------------

def pdf_page_to_gray(pdf_path: str, page_index: int) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
        sys.exit("PyMuPDF not found. Install it with:  pip install pymupdf")

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        sys.exit(f"Page {page_index} out of range — PDF has {len(doc)} pages.")

    page = doc[page_index]
    mat = fitz.Matrix(DPI / 72, DPI / 72)   # 72 pt/inch → target DPI
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return arr


# ---------------------------------------------------------------------------
# Step 2 — Clean binary image
# ---------------------------------------------------------------------------

def binarise(gray: np.ndarray) -> np.ndarray:
    # Salt-and-pepper noise from scanning
    denoised = cv2.medianBlur(gray, ksize=3)

    # Gaussian adaptive threshold handles uneven lighting across the scanned page
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # ink → white, background → black
        blockSize=31,
        C=10,
    )
    return binary


# ---------------------------------------------------------------------------
# Step 3 — Detect configuration bounding boxes
# ---------------------------------------------------------------------------

# TODO: refine bounding box boundarys
def find_config_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) bounding boxes for each detected configuration cell."""

    # **dilate** to merge nearby strokes that belong to the same diagram
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # find contours, discarding noise
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    # filter contours by size and aspect ratio
    #h_page, w_page = binary.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CELL_AREA < area < MAX_CELL_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Label-only false positives are ~70px tall; real diagrams are 200px+
        if h < 150:
            continue

        aspect = w / h if h > 0 else 0
        if not (ASPECT_RATIO_BOUNDS[0] < aspect < ASPECT_RATIO_BOUNDS[1]):
            continue

        boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right (reading order)
    boxes.sort(key=lambda b: (b[1] // 100, b[0]))
    return boxes


# ---------------------------------------------------------------------------
# Step 4 — Crop each box out of the original grayscale image
# ---------------------------------------------------------------------------

def crop_configs(gray: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    h_img, w_img = gray.shape
    crops = []
    for x, y, w, h in boxes:
        x0 = max(0, x - PADDING)
        y0 = max(0, y - PADDING)
        x1 = min(w_img, x + w + PADDING)
        y1 = min(h_img, y + h + PADDING)
        crops.append(gray[y0:y1, x0:x1])
    return crops


# ---------------------------------------------------------------------------
# Step 5 — Perspective correction (optional, applied when page is skewed)
# ---------------------------------------------------------------------------

def maybe_deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten the page if it was scanned at a slight angle."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:   # negligible skew
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_page(pdf_path: str, page_index: int, out_dir: str, debug: bool = False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading page {page_index} from {pdf_path} at {DPI} DPI...")
    gray = pdf_page_to_gray(pdf_path, page_index)
    gray = maybe_deskew(gray)

    print("Binarising...")
    binary = binarise(gray)

    print("Detecting configuration cells...")
    boxes = find_config_boxes(binary)
    print(f"  Found {len(boxes)} candidate cells.")

    crops = crop_configs(gray, boxes)
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        fname = out_path / f"page{page_index:03d}_cell{i:03d}.png"
        cv2.imwrite(str(fname), crop)
        print(f"  Saved {fname}  (box: x={box[0]} y={box[1]} w={box[2]} h={box[3]})")

    if debug:
        # Annotated overview saved alongside the crops
        overview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(overview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_path = out_path / f"page{page_index:03d}_debug.png"
        cv2.imwrite(str(debug_path), overview)
        print(f"  Debug overview saved to {debug_path}")

    return crops, boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: extract config crops from PDF page")
    parser.add_argument("--pdf",  default=DEFAULT_PDF, help="Path to Appel-Haken PDF")
    parser.add_argument("--page", type=int, default=0, help="0-based page index")
    parser.add_argument("--out",  default="../data/crops", help="Output directory for cropped images")
    parser.add_argument("--debug", action="store_true", help="Save annotated overview image")
    args = parser.parse_args()

    process_page(args.pdf, args.page, args.out, debug=args.debug)