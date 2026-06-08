"""
Phase 2 — Node Detection

Detects and classifies nodes in a cropped Appel-Haken configuration image.

Node types (from CLAUDE.md):
    solid_dot   → degree 5  (large filled black disc)
    open_circle → degree 6  (hollow ring, white interior)
    square      → degree 7  (four-cornered blob)
    triangle    → degree 8+ (three-cornered blob)

Strategy:
    Solid dots are filled blobs — detected via connected components + circularity.
    Open circles are hollow rings — detected via Hough on the ring edge.
    Squares / triangles — detected via contour polygon approximation.

Usage:
    python node_detector.py crops/page043_cell002.png
    python node_detector.py crops/page043_cell002.png --debug
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tuning constants (calibrated at 600 DPI on ~370–560px-wide crops)
# ---------------------------------------------------------------------------

# Fraction of image height stripped from bottom to remove "CTL #N" label
LABEL_STRIP_FRAC = 0.18

# Adaptive threshold block size (should be odd; ~5% of crop width at 600 DPI)
BINARISE_BLOCK = 31

# --- Solid dot detection via morphological erosion ---
# At 600 DPI: edges ~8-10px wide, solid dot radius ~10px.
# Eroding with r=8 strips edges (leaves area<40) but solid dot cores survive (area>50).
SOLID_ERODE_R = 8
SOLID_CORE_MIN_AREA = 50   # px² after erosion — anything smaller is an edge fragment
SOLID_CORE_MAX_AREA = 800  # px² — cap to avoid merged regions

# --- Open circle (Hough ring) detection ---
# At 600 DPI, open circle rings have radius ~12–20px
HOUGH_MIN_RADIUS = 10
HOUGH_MAX_RADIUS = 22
HOUGH_MIN_DIST = 28
HOUGH_THRESHOLD = 18
OPEN_MAX_FILL = 0.38           # interior must be mostly white

# --- Square / triangle (polygon) detection ---
POLY_MIN_AREA = 400
POLY_MAX_AREA = 5_000
POLY_MAX_CIRCULARITY = 0.72    # below this → non-circular blob

# Proximity radius: skip a candidate if a previously found node is already near
PROXIMITY_PX = 25

DEGREE_MAP = {
    "solid_dot":   5,
    "open_circle": 6,
    "square":      7,
    "triangle":    8,
}


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class Node:
    x: int
    y: int
    radius: int
    shape: str       # solid_dot | open_circle | square | triangle
    degree: int


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_binary(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray, binary_inv) with the bottom label strip removed."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h = gray.shape[0]
    gray = gray[:h - int(h * LABEL_STRIP_FRAC), :]

    denoised = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=BINARISE_BLOCK, C=8,
    )
    return gray, binary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_ratio(binary_inv: np.ndarray, cx: int, cy: int, r: int,
                sample_frac: float = 0.70) -> float:
    """Fraction of pixels within sample_frac*r of (cx,cy) that are ink."""
    h, w = binary_inv.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r * sample_frac) ** 2
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return int(binary_inv[mask].sum() / 255) / total


def _near_any(cx: int, cy: int, nodes: list[Node], threshold: int = PROXIMITY_PX) -> bool:
    return any((cx - n.x) ** 2 + (cy - n.y) ** 2 < threshold ** 2 for n in nodes)


# ---------------------------------------------------------------------------
# Step 1 — Solid dot detection (morphological erosion)
# ---------------------------------------------------------------------------
#
# TODO: solid dot detection is incomplete and needs further work.
# Several approaches were attempted and abandoned:
#   - Hough circles: couldn't distinguish filled discs from hollow rings or
#     face boundaries; all circles were being classified as open_circle.
#   - Matched filter (circular convolution): dense triangulation means face
#     interiors score as high as node centres — no clean threshold exists.
#   - Distance transform peaks: max dist at 600 DPI is only ~10px, giving a
#     gap of <1px between solid dot peaks and edge-midpoint peaks — unreliable.
#   - Fat-ink components (threshold dist transform): blob circularity is ~0.08
#     at junctions (star-shaped), so circularity filters reject real nodes.
# Current approach (erosion r=8) works on some configurations but misses solid
# dots in dense graphs where the morphological gap is too narrow. The HITL UI
# is the designed correction path for missed/misclassified nodes.

def detect_solid_dots(binary_inv: np.ndarray) -> list[Node]:
    """
    Erode with SOLID_ERODE_R to strip thin edges; solid dot cores survive.
    At 600 DPI: edge peaks ≤ ~9px, solid dot peaks ~10px — eroding at r=8
    removes edge stripes (residual area <40px²) while solid dot cores remain
    (area >50px²). Use the dist-transform peak of each core as the true centre.
    """
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (SOLID_ERODE_R * 2 + 1,) * 2)
    eroded = cv2.erode(binary_inv, erode_k, iterations=1)
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded)
    nodes: list[Node] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if not (SOLID_CORE_MIN_AREA <= area <= SOLID_CORE_MAX_AREA):
            continue
        comp_mask = labels == label
        peak_val = float(dist[comp_mask].max())
        peak_ys, peak_xs = np.where(comp_mask & (dist >= peak_val - 0.5))
        cx = int(peak_xs.mean())
        cy = int(peak_ys.mean())
        r = max(1, int(peak_val))
        nodes.append(Node(x=cx, y=cy, radius=r, shape="solid_dot", degree=5))

    return nodes


# ---------------------------------------------------------------------------
# Step 2 — Open circle detection (Hough on the ring edge)
# ---------------------------------------------------------------------------

def detect_open_circles(gray: np.ndarray, binary_inv: np.ndarray,
                        existing: list[Node]) -> list[Node]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=HOUGH_MIN_DIST,
        param1=50,
        param2=HOUGH_THRESHOLD,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )

    nodes: list[Node] = []
    if circles is None:
        return nodes

    for cx, cy, r in np.round(circles[0]).astype(int):
        cx, cy, r = int(cx), int(cy), int(r)
        if _near_any(cx, cy, existing):
            continue
        # TODO: solid/open classification via fill ratio is unreliable.
        # Hough finds circles at the outer boundary of nodes, so the detected
        # centre can sit over white space between edges rather than over the ink
        # core — especially for solid dots at dense junctions. Reducing
        # sample_frac to 0.30 (from 0.70) helped avoid edge bleed but the
        # thresholds (>0.60 = solid, <0.38 = open) are still fragile across
        # configurations. A better approach may be to use the grayscale image
        # directly (mean intensity at centre) rather than the binarised image.
        fill = _fill_ratio(binary_inv, cx, cy, r, sample_frac=0.30)
        if fill > 0.60:
            nodes.append(Node(x=cx, y=cy, radius=r, shape="solid_dot", degree=5))
        elif fill < OPEN_MAX_FILL:
            nodes.append(Node(x=cx, y=cy, radius=r, shape="open_circle", degree=6))
        # 0.38–0.60 is ambiguous (face annotation numerals etc.) — discard

    return nodes


# ---------------------------------------------------------------------------
# Step 3 — Square / triangle detection (polygon contours)
# ---------------------------------------------------------------------------

def detect_polygon_nodes(binary_inv: np.ndarray, existing: list[Node]) -> list[Node]:
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes: list[Node] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (POLY_MIN_AREA <= area <= POLY_MAX_AREA):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= POLY_MAX_CIRCULARITY:
            continue  # circular — handled above

        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        corners = len(approx)
        if corners == 4:
            shape = "square"
        elif corners == 3:
            shape = "triangle"
        else:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if _near_any(cx, cy, existing):
            continue

        r = max(1, int(np.sqrt(area / np.pi)))
        nodes.append(Node(x=cx, y=cy, radius=r, shape=shape, degree=DEGREE_MAP[shape]))

    return nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_nodes(image_path: str) -> list[Node]:
    gray, binary_inv = load_binary(image_path)
    solid_dots = detect_solid_dots(binary_inv)
    open_circles = detect_open_circles(gray, binary_inv, solid_dots)
    polygons = detect_polygon_nodes(binary_inv, solid_dots + open_circles)
    return solid_dots + open_circles + polygons


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

COLOURS = {
    "solid_dot":   (0,   0,   255),   # red
    "open_circle": (255, 140,   0),   # orange
    "square":      (0,   200,   0),   # green
    "triangle":    (200,   0, 200),   # purple
}


def draw_nodes(image_path: str, nodes: list[Node], out_path: str | None = None) -> np.ndarray:
    img = cv2.imread(image_path)
    for n in nodes:
        colour = COLOURS[n.shape]
        cv2.circle(img, (n.x, n.y), n.radius + 4, colour, 2)
        cv2.putText(img, str(n.degree), (n.x - 6, n.y - n.radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, img)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: detect nodes in a config crop")
    parser.add_argument("image", help="Path to crop PNG")
    parser.add_argument("--debug", action="store_true", help="Save annotated image alongside input")
    args = parser.parse_args()

    nodes = detect_nodes(args.image)

    print(f"\n{Path(args.image).name}  —  {len(nodes)} node(s) detected")
    print(f"  {'shape':<14} {'deg':>4}  {'(x, y)':>14}  {'r':>4}")
    print(f"  {'-'*14} {'----':>4}  {'------':>14}  {'--':>4}")
    for n in nodes:
        print(f"  {n.shape:<14} {n.degree:>4}  ({n.x:>5}, {n.y:>5})  {n.radius:>4}")

    if args.debug:
        p = Path(args.image)
        out = str(p.parent / (p.stem + "_nodes.png"))
        draw_nodes(args.image, nodes, out)
        print(f"\nDebug image saved to {out}")