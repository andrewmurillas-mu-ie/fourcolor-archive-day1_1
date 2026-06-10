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

# --- Solid dot detection via distance-transform local maxima ---
# At 600 DPI: edge half-width ~4–6 px (dist peak ~4–6), solid dot radius ~10–25 px
# (dist peak ~10–25).  Threshold at 10 px rejects thick-edge midpoints (dist ~8–9)
# while keeping genuine solid dots.  Fill ratio 0.65 further guards against
# elongated junction blobs that survive the distance threshold.
SOLID_MIN_DIST_PX = 10.0   # minimum dist-transform value to qualify as a node centre
SOLID_FILL_MIN    = 0.65   # fill ratio within 0.5*r must exceed this

# --- Open circle (Hough ring) detection ---
# At 600 DPI, open circle rings have radius ~12–20px.
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


def _radial_ink_count(binary_inv: np.ndarray, cx: int, cy: int, r: int,
                      n_angles: int = 8, factor: float = 1.5) -> int:
    """Count how many of n_angles directions have ink at distance factor*r.

    Real nodes have edges attached, so several outward directions are ink.
    Annotation blobs sitting in an empty face have 0 ink at that distance.
    """
    sample_r = r * factor
    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    xs = np.clip((cx + sample_r * np.cos(angles)).astype(int), 0, binary_inv.shape[1] - 1)
    ys = np.clip((cy + sample_r * np.sin(angles)).astype(int), 0, binary_inv.shape[0] - 1)
    return int((binary_inv[ys, xs] > 128).sum())


# ---------------------------------------------------------------------------
# Step 1 — Solid dot detection (distance-transform local maxima)
# ---------------------------------------------------------------------------

def detect_solid_dots(binary_inv: np.ndarray) -> list[Node]:
    """
    Each foreground pixel's distance-transform value = distance to the nearest
    background pixel.  At the centre of a solid dot (~10–25 px radius at
    600 DPI) this value is ~10–25 px; at an edge midpoint (~4–6 px half-width)
    it is ~4–6 px.  Thresholding at SOLID_MIN_DIST_PX=8 cleanly separates
    them without a fragile erosion radius or area cap.
    """
    # Close small ink gaps (≤9 px) so the narrow white spaces between edge arms
    # at dense hubs don't suppress the hub's dist-transform peak.  The original
    # binary_inv is kept for fill-ratio classification so we don't inflate scores
    # on non-node regions that closing artificially fills.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, close_k)
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)

    # Non-maximum suppression: keep pixels whose dist value is the local max
    # within a window of radius PROXIMITY_PX (one peak per node spacing).
    ksize = PROXIMITY_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    local_max = cv2.dilate(dist, kernel)
    peak_mask = (
        (dist >= local_max - 0.5) & (dist >= SOLID_MIN_DIST_PX)
    ).astype(np.uint8)

    # Connected components to merge any adjacent equal-value peak pixels.
    num, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask)

    nodes: list[Node] = []
    for lbl in range(1, num):
        comp_mask = labels == lbl
        peak_val = float(dist[comp_mask].max())
        peak_ys, peak_xs = np.where(comp_mask & (dist >= peak_val - 0.5))
        cx = int(peak_xs.mean())
        cy = int(peak_ys.mean())
        r = max(1, int(peak_val))

        # Sample close to the centre (0.3r) so edge arms radiating from large
        # hubs don't reduce the fill ratio into the rejection zone.
        fill = _fill_ratio(binary_inv, cx, cy, r, sample_frac=0.3)
        if fill < SOLID_FILL_MIN:
            continue

        # Annotation blobs (boosted by closing) sit in empty faces — they have
        # no ink in any outward direction at 1.5r.  Real nodes have ≥2 attached
        # edges visible at that distance.
        if _radial_ink_count(binary_inv, cx, cy, r) < 2:
            continue

        if _near_any(cx, cy, nodes):
            continue

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