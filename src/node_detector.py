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

# Adaptive threshold block sizes (should be odd)
BINARISE_BLOCK       = 31   # used for edge/open-circle features (fine detail)
SOLID_BINARISE_BLOCK = 61   # used for solid dot detection only — larger window
                             # prevents adaptive threshold from hollowing out large
                             # solid discs (block=31 makes their centres appear white)

# --- Solid dot detection via distance-transform local maxima ---
# At 600 DPI: edge half-width ~4–6 px (dist peak ~4–6), solid dot radius ~10–25 px
# (dist peak ~10–25).  Threshold at 10 px rejects thick-edge midpoints (dist ~8–9)
# while keeping genuine solid dots.  Fill ratio 0.65 further guards against
# elongated junction blobs that survive the distance threshold.
SOLID_MIN_DIST_PX  = 10.0   # minimum dist-transform value to qualify as a node centre
SOLID_FILL_MIN     = 0.65   # fill ratio within 0.3*r must exceed this
SOLID_LARGE_RADIUS = 14     # nodes with r >= this only need 1 outward ink direction
                             # (both edge arms may fall in a narrow angular range)

# --- Open circle (Hough ring) detection ---
# At 600 DPI, open circle rings have radius ~12–20px.
HOUGH_MIN_RADIUS = 10
HOUGH_MAX_RADIUS = 22
HOUGH_MIN_DIST = 28
HOUGH_THRESHOLD = 18
OPEN_MIN_CENTRE_INTENSITY = 160  # mean grayscale in centre must exceed this (0=black, 255=white)
OPEN_MIN_RING_INK = 0.45         # fraction of circumference at radius r that must be ink

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
    """One detected vertex in a configuration crop.

    Attributes:
        x, y:    centre in crop pixel coordinates (600 DPI).
        radius:  marker radius in pixels (used for probe skip zones and UI).
        shape:   Heesch symbol: solid_dot | open_circle | square | triangle.
        degree:  specified degree implied by the shape (see DEGREE_MAP).
    """
    x: int
    y: int
    radius: int
    shape: str       # solid_dot | open_circle | square | triangle
    degree: int


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_denoised(image_path: str) -> np.ndarray:
    """Load, strip label row, and median-denoise. Returns denoised gray."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h = gray.shape[0]
    return cv2.medianBlur(gray[:h - int(h * LABEL_STRIP_FRAC), :], 3)


def _binarise(denoised: np.ndarray, block_size: int) -> np.ndarray:
    """Adaptive-threshold to an ink mask (ink=255, paper=0).

    block_size controls the Gaussian window: 31 preserves thin lines and
    open-circle rims; 61 keeps large solid discs filled (see
    SOLID_BINARISE_BLOCK comment above).
    """
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size, C=8,
    )


def load_binary(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (denoised_gray, binary_inv) with the bottom label strip removed."""
    denoised = _load_denoised(image_path)
    return denoised, _binarise(denoised, BINARISE_BLOCK)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_ratio(binary_inv: np.ndarray, cx: int, cy: int, r: int,
                sample_frac: float | int = 0.70) -> float | int:
    """Fraction of pixels within sample_frac*r of (cx,cy) that are ink."""
    h, w = binary_inv.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r * sample_frac) ** 2
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return int(binary_inv[mask].sum() / 255) / total


def _near_any(cx: int, cy: int, nodes: list[Node], threshold: int = PROXIMITY_PX) -> bool:
    return any((cx - node.x) ** 2 + (cy - node.y) ** 2 < threshold ** 2 for node in nodes)


def _mean_intensity(gray: np.ndarray, cx: int, cy: int, r: int,
                    sample_frac: float | int = 0.4) -> float | int:
    """Mean grayscale value within sample_frac*r of (cx, cy). High = white/hollow centre."""
    h, w = gray.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r * sample_frac) ** 2
    pixels = gray[mask]
    return float(pixels.mean()) if pixels.size else 255.0


def _ring_ink_density(binary_inv: np.ndarray, cx: int, cy: int, r: int,
                      n_angles: int = 24) -> float | int:
    """Fraction of circumference points at radius r that are ink.

    Real open circle nodes have their ring boundary as actual ink, so most
    sampled points hit the ring.  Hough false positives fitted to the white
    gap between spokes at a hub have white space between spoke directions,
    giving a low fraction.
    """
    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, binary_inv.shape[1] - 1)
    ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, binary_inv.shape[0] - 1)
    return float((binary_inv[ys, xs] > 128).sum()) / n_angles


def _radial_ink_count(binary_inv: np.ndarray, cx: int, cy: int, r: int,
                      n_angles: int = 8, factor: float | int = 1.5) -> int:
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

def detect_solid_dots(binary_inv: np.ndarray,
                       binary_inv_solid: np.ndarray) -> list[Node]:
    """
    Each foreground pixel's distance-transform value = distance to the nearest
    background pixel.  At the centre of a solid dot (~10–25 px radius at
    600 DPI) this value is ~10–25 px; at an edge midpoint (~4–6 px half-width)
    it is ~4–6 px.

    binary_inv_solid (block=61) is used for the distance transform and fill
    check — it correctly fills large solid discs that block=31 hollows out.
    binary_inv (block=31) is used for the radial ink count — it has better
    resolution for thin edge lines.
    """
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary_inv_solid, cv2.MORPH_CLOSE, close_k)
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)

    ksize = PROXIMITY_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    local_max = cv2.dilate(dist, kernel)
    peak_mask = (
        (dist >= local_max - 0.5) & (dist >= SOLID_MIN_DIST_PX)
    ).astype(np.uint8)

    num, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask)

    nodes: list[Node] = []
    for lbl in range(1, num):
        comp_mask = labels == lbl
        peak_val = float(dist[comp_mask].max())
        peak_ys, peak_xs = np.where(comp_mask & (dist >= peak_val - 0.5))
        cx = int(peak_xs.mean())
        cy = int(peak_ys.mean())
        r = max(1, int(peak_val))

        fill = _fill_ratio(binary_inv_solid, cx, cy, r, sample_frac=0.3)
        if fill < SOLID_FILL_MIN:
            continue

        # 16 angles give better angular coverage than 8 for nodes with edges
        # bunched in a narrow arc.  Large nodes (r >= SOLID_LARGE_RADIUS) only
        # need 1 hit — both attached edges may point in similar directions.
        radial = _radial_ink_count(binary_inv, cx, cy, r, n_angles=16)
        min_radial = 1 if r >= SOLID_LARGE_RADIUS else 2
        if radial < min_radial:
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
    """Hough-circle pass for degree-6 hollow rings: candidates must have a
    bright centre, ink on the rim, and outward ink (attached edges);
    candidates near already-found nodes are suppressed."""
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
        if _mean_intensity(gray, cx, cy, r) < OPEN_MIN_CENTRE_INTENSITY:
            continue
        if _radial_ink_count(binary_inv, cx, cy, r) < 2:
            continue
        if _ring_ink_density(binary_inv, cx, cy, r) < OPEN_MIN_RING_INK:
            continue
        nodes.append(Node(x=cx, y=cy, radius=r, shape="open_circle", degree=6))

    return nodes


# ---------------------------------------------------------------------------
# Step 3 — Square / triangle detection (polygon contours)
# ---------------------------------------------------------------------------

def detect_polygon_nodes(binary_inv: np.ndarray, existing: list[Node]) -> list[Node]:
    """Contour pass for squares (degree 7) and triangles (degree 8): low
    circularity blobs classified by approxPolyDP corner count."""
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
    """Detect and classify all vertices in a configuration crop.

    Three passes in fixed priority order, each suppressing candidates within
    PROXIMITY_PX of an earlier hit: solid dots (distance-transform peaks),
    open circles (Hough on the rim), then squares/triangles (polygon
    contours).  Returns Nodes in detection order — this index order is the
    node id used by edge lists and downstream JSON.
    """
    gray, binary_inv = load_binary(image_path)
    binary_inv_solid = _binarise(gray, SOLID_BINARISE_BLOCK)
    solid_dots = detect_solid_dots(binary_inv, binary_inv_solid)
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
    """Render detections over the crop (colour-coded by shape, labelled with
    degree).  Writes to out_path when given; returns the annotated image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    for node in nodes:
        colour = COLOURS[node.shape]
        cv2.circle(img, (node.x, node.y), node.radius + 4, colour, 2)
        cv2.putText(img, str(node.degree), (node.x - 6, node.y - node.radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, img)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: detect nodes in one crop and print a table (--debug saves an
    annotated image alongside the input)."""
    parser = argparse.ArgumentParser(description="Phase 2: detect nodes in a config crop")
    parser.add_argument("image", help="Path to crop PNG")
    parser.add_argument("--debug", action="store_true", help="Save annotated image alongside input")
    args = parser.parse_args()

    nodes = detect_nodes(args.image)

    print(f"\n{Path(args.image).name}  —  {len(nodes)} node(s) detected")
    print(f"  {'shape':<14} {'deg':>4}  {'(x, y)':>14}  {'r':>4}")
    print(f"  {'-'*14} {'----':>4}  {'------':>14}  {'--':>4}")
    for node in nodes:
        print(f"  {node.shape:<14} {node.degree:>4}  ({node.x:>5}, {node.y:>5})  {node.radius:>4}")

    if args.debug:
        p = Path(args.image)
        out = str(p.parent / (p.stem + "_nodes.png"))
        draw_nodes(args.image, nodes, out)
        print(f"\nDebug image saved to {out}")


if __name__ == "__main__":
    main()