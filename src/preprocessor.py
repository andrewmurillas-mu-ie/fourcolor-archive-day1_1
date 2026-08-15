"""
Phase 1 — Image Pre-processing

Converts pages of the Appel-Haken PDF into clean, cropped sub-images,
one per configuration diagram.

Usage:
    python preprocessor.py                         # page 0 only
    python preprocessor.py --page 5               # single page (0-based)
    python preprocessor.py --pages 14-120         # inclusive range
    python preprocessor.py --pages 14-120 --debug # also save annotated overview per page
    python preprocessor.py --pdf path/to.pdf      # custom PDF path
    python preprocessor.py --out crops/           # output directory for crops

Requires:  pip install pymupdf pytesseract pillow
           Also needs Tesseract on the system (brew install tesseract)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 600          # 600 DPI doubles all pixel distances vs 300 DPI, giving a clear
                   # morphological gap between thin edges (~10px) and solid dots (~20px)
MIN_CELL_AREA = 8_000   # px² — discard tiny noise contours
MAX_CELL_AREA = 1_000_000  # px² — discard full-page contour
ASPECT_RATIO_BOUNDS = (0.4, 2.5)  # width/height range for a plausible config cell
PADDING = 10        # px to add around each crop
LABEL_SEARCH_GAP = 300   # px below graph bottom to look for its C/D letter

# ---------------------------------------------------------------------------
# Step 1 — PDF page → grayscale NumPy array
# ---------------------------------------------------------------------------

def pdf_page_to_gray(pdf_path: str, page_index: int) -> np.ndarray:
    """Render one PDF page (0-based index) to a grayscale array at DPI=600.

    Returns an empty array if page_index is past the end of the document.
    600 DPI is load-bearing: every pixel constant in node_detector.py and
    edge_detector.py is calibrated to it.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
        sys.exit("PyMuPDF not found. Install it with:  pip install pymupdf")

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        print(f"  Skipping page {page_index} — PDF only has {len(doc)} pages.")
        return np.array([], dtype=np.uint8)
    page = doc[page_index]
    mat = fitz.Matrix(DPI / 72, DPI / 72)   # 72 pt/inch → target DPI
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return arr


# ---------------------------------------------------------------------------
# Step 2 — Clean binary image
# ---------------------------------------------------------------------------

def binarise(gray: np.ndarray) -> np.ndarray:
    """Median-denoise + Gaussian adaptive threshold -> ink mask (ink=255)."""
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

    # dilate to merge strokes within the same diagram without merging adjacent diagrams
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
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
    """Cut each detected box out of the page with PADDING px on every side
    (clamped at page borders).  Crop k corresponds to box k in order."""
    h_img, w_img = gray.shape
    config_crops = []
    for x, y, w, h in boxes:
        x0 = max(0, x - PADDING)
        y0 = max(0, y - PADDING)
        x1 = min(w_img, x + w + PADDING)
        y1 = min(h_img, y + h + PADDING)
        config_crops.append(gray[y0:y1, x0:x1])
    return config_crops

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
    """Full Phase-1 pipeline for one page: render -> deskew -> binarise ->
    detect cells -> save crops as page{p:03d}_cell{k:03d}.png.

    DETERMINISM CONTRACT: given the same PDF and page, this always produces
    identical boxes in identical order.  src/ring_labels.py relies on it to
    map existing crops back to Table U slots without re-reading the crops.
    Returns (crops, boxes).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading page {page_index} from {pdf_path} at {DPI} DPI...")
    grayed_pdf = pdf_page_to_gray(pdf_path, page_index)
    if grayed_pdf.size == 0:
        return [], []
    grayed_pdf = maybe_deskew(grayed_pdf)

    print("Binarising...")
    binary = binarise(grayed_pdf)

    print("Detecting configuration cells...")
    boxes = find_config_boxes(binary)
    print(f"  Found {len(boxes)} candidate cells.")

    crops = crop_configs(grayed_pdf, boxes)

    for i, (crop, box) in enumerate(zip(crops, boxes)):

        # Save graph crop to /crops
        sub_dir = out_path
        sub_dir.mkdir(parents=True, exist_ok=True)
        fname = sub_dir / f"page{page_index:03d}_cell{i:03d}.png"
        cv2.imwrite(str(fname), crop)
        print(f"  Saved {fname}  (box: x={box[0]} y={box[1]} w={box[2]} h={box[3]})")

    if debug:
        # Annotated overview saved alongside the crops
        overview = cv2.cvtColor(grayed_pdf, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(overview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_path = out_path / f"page{page_index:03d}_debug.png"
        cv2.imwrite(str(debug_path), overview)
        print(f"  Debug overview saved to {debug_path}")

    return crops, boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: extract config crops from PDF pages")
    parser.add_argument("--pdf",   default=os.getenv("DEFAULT_PDF"), help="Path to Appel-Haken PDF")
    parser.add_argument("--page",  type=int, default=None, help="Single 0-based page index")
    parser.add_argument("--pages", default=None,
                        help="Inclusive page range, e.g. 14-120")
    parser.add_argument("--out",   default="../data/crops", help="Output directory for crops")
    parser.add_argument("--debug", action="store_true", help="Save annotated overview per page")
    args = parser.parse_args()

    if args.pages:
        try:
            start, end = (int(x) for x in args.pages.split("-"))
        except ValueError:
            sys.exit("--pages must be in the form START-END, e.g. 14-120")
        page_list = range(start, end + 1)
    elif args.page is not None:
        page_list = [args.page]
    else:
        page_list = range(14, 121)

    total_crops = 0
    for pg in page_list:
        pg_crops, _ = process_page(args.pdf, pg, args.out, debug=args.debug)
        total_crops += len(pg_crops)

    if len(page_list) > 1:
        print(f"\nTotal crops saved: {total_crops} across {len(page_list)} pages.")