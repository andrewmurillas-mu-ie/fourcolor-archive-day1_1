"""
Phase 4 — HITL Correction UI

Displays a 600 DPI crop with auto-detected nodes and edges overlaid.
The user corrects detections interactively; the Euler identity validator
runs after every change; a clean JSON is saved when it passes.

Controls
--------
  Left-click node          cycle shape  (solid_dot → open_circle → square → triangle)
  Shift + left-click node  delete node
  Left-click empty space   add a solid_dot node at that position
  Left-click edge          delete edge
  A key                    enter Add-Edge mode (click two nodes to connect)
  Esc                      cancel Add-Edge mode
  + / -                    increase / decrease ring size  r
  ] / [                    increase / decrease E_attachment
  S key                    save to JSON  (validator must pass first)
  N / P keys               next / previous crop  (when launched on a directory)

Usage
-----
  python hitl_ui.py crops_part2/page014_cell005.png
  python hitl_ui.py crops_part2/                         # browse all PNGs
  python hitl_ui.py crops_part2/ --detections detections_part2.json
  python hitl_ui.py crops_part2/ --detections detections_part2.json --skip-empty
  python hitl_ui.py crops_part2/ --detections detections_part2.json --out annotations/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Any
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from node_detector import Node, detect_nodes, load_binary, DEGREE_MAP
from validator import Validator

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SHAPE_CYCLE = ["solid_dot", "open_circle", "square", "triangle"]

COLOURS = {
    "solid_dot":   "#e74c3c",   # red
    "open_circle": "#e67e22",   # orange
    "square":      "#27ae60",   # green
    "triangle":    "#8e44ad",   # purple
}

HIT_RADIUS_PX  = 18   # pixel distance to count as "clicking on" a node
EDGE_HIT_PX    = 10   # pixel distance to count as "clicking on" an edge
LINE_PROBE_THRESH = 0.45   # fraction of sampled pixels that must be ink
# Skip is now computed per-pair from node radii (see _probe_edge); this
# constant is kept only as a minimum floor.
LINE_PROBE_SKIP_MIN = 6

_validator = Validator()


# ──────────────────────────────────────────────────────────────────────────────
# Edge auto-detection (line probe)
# ──────────────────────────────────────────────────────────────────────────────

def _probe_edge(binary_inv: np.ndarray, n1: Node, n2: Node) -> bool:
    """Return True if enough dark pixels lie along the line between n1 and n2."""
    x1, y1, x2, y2 = n1.x, n1.y, n2.x, n2.y
    length = float(np.hypot(x2 - x1, y2 - y1))
    # Skip at least the node's own radius so we don't sample through the node disc.
    skip = max(LINE_PROBE_SKIP_MIN, n1.radius, n2.radius)
    if length < 2 * skip + 4:
        return False

    t = np.linspace(0.0, 1.0, int(length) + 1)
    skip_frac = skip / length
    valid = (t > skip_frac) & (t < 1.0 - skip_frac)

    xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, binary_inv.shape[1] - 1)
    ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, binary_inv.shape[0] - 1)

    xs_v, ys_v = xs[valid], ys[valid]
    if len(xs_v) == 0:
        return False

    ink = (binary_inv[ys_v, xs_v] > 128).mean()
    return float(ink) >= LINE_PROBE_THRESH


def probe_all_edges(binary_inv: np.ndarray, nodes: list[Node]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _probe_edge(binary_inv, nodes[i], nodes[j]):
                edges.add((i, j))
    return edges


# ──────────────────────────────────────────────────────────────────────────────
# Main editor class
# ──────────────────────────────────────────────────────────────────────────────

class HITLEditor:
    def __init__(self, image_path: str, out_dir: str = "annotations",
                 preloaded_nodes: list[Node] | None = None):
        self.image_path = Path(image_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load image ──────────────────────────────────────────────────────
        bgr = cv2.imread(str(self.image_path))
        if bgr is None:
            sys.exit(f"Cannot read {self.image_path}")
        self.display_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Binary image (label-stripped) used for line-probe edge detection
        _, self.binary_inv = load_binary(str(self.image_path))

        # ── Graph state ─────────────────────────────────────────────────────
        self.nodes: list[Node] = (
            preloaded_nodes if preloaded_nodes is not None
            else detect_nodes(str(self.image_path))
        )
        self.edges: set[tuple[int, int]] = probe_all_edges(self.binary_inv, self.nodes)
        self.r: int = 5          # ring size — user adjustable
        self.e_attach: int = 0   # E_attachment count — user adjustable

        # ── UI state ────────────────────────────────────────────────────────
        self.mode: str = "normal"   # "normal" | "add_edge"
        self.edge_first: int | None = None   # first node index during add_edge

        # ── Build figure ────────────────────────────────────────────────────
        self._build_figure()

        self.redraw()
        plt.show()

    # ── Figure layout ────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.patch.set_facecolor("#1e1e2e")

        # Image axes (left 68%)
        self.ax: Axes = self.fig.add_axes((0.01, 0.08, 0.66, 0.90))
        self.ax.set_facecolor("#1e1e2e")
        self.ax.axis("off")

        # Info panel (right 30%)
        self.info_ax = self.fig.add_axes((0.69, 0.08, 0.30, 0.90))
        self.info_ax.set_facecolor("#2a2a3e")
        self.info_ax.axis("off")

        # Button bar (bottom strip)
        from matplotlib.widgets import Button

        ax_add  = self.fig.add_axes((0.02,  0.01, 0.18, 0.055))
        ax_save = self.fig.add_axes((0.22,  0.01, 0.18, 0.055))
        ax_next = self.fig.add_axes((0.44,  0.01, 0.10, 0.055))
        ax_prev = self.fig.add_axes((0.56,  0.01, 0.10, 0.055))

        self.btn_add  = Button(ax_add,  "Add Edge [A]",  color="#3a3a5c", hovercolor="#5555aa")
        self.btn_save = Button(ax_save, "Save JSON [S]", color="#2d4a2d", hovercolor="#3a6b3a")
        self.btn_next = Button(ax_next, "Next [N]",      color="#3a3a5c", hovercolor="#5555aa")
        self.btn_prev = Button(ax_prev, "Prev [P]",      color="#3a3a5c", hovercolor="#5555aa")

        for btn in (self.btn_add, self.btn_save, self.btn_next, self.btn_prev):
            btn.label.set_color("white")
            btn.label.set_fontsize(9)

        self.btn_add.on_clicked(lambda _: self._toggle_add_edge())
        self.btn_save.on_clicked(lambda _: self._save())
        self.btn_next.on_clicked(lambda _: self._advance(+1))
        self.btn_prev.on_clicked(lambda _: self._advance(-1))

        # Events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def redraw(self):
        self.ax.cla()
        self.ax.set_facecolor("#1e1e2e")
        self.ax.axis("off")

        # Background image
        self.ax.imshow(self.display_img, origin="upper", aspect="equal")
        h, w = self.display_img.shape[:2]
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)

        # Edges
        for i, j in self.edges:
            n1, n2 = self.nodes[i], self.nodes[j]
            self.ax.plot([n1.x, n2.x], [n1.y, n2.y],
                         color="#00ffe0", linewidth=1.5, alpha=0.7, zorder=2)

        # Nodes
        for idx, n in enumerate(self.nodes):
            c = COLOURS[n.shape]
            # Highlight first-selected node in add-edge mode
            lw = 3.5 if idx == self.edge_first else 2.0
            ring_r = n.radius + 5
            circle = patches.Circle((n.x, n.y), ring_r,
                                    edgecolor=c, facecolor="none",
                                    linewidth=lw, zorder=3)
            self.ax.add_patch(circle)
            self.ax.text(n.x, n.y - ring_r - 5, str(n.degree),
                         color=c, fontsize=7, ha="center", va="bottom",
                         fontweight="bold", zorder=4)

        # Title
        self.ax.set_title(
            self.image_path.name,
            color="white", fontsize=9, pad=4
        )

        # Info panel
        self._draw_info()

        self.fig.canvas.draw_idle()

    def _draw_info(self):
        self.info_ax.cla()
        self.info_ax.set_facecolor("#2a2a3e")
        self.info_ax.axis("off")

        result = _validator.check(
            V          = len(self.nodes) + self.r,
            E_internal = len(self.edges),
            E_attachment = self.e_attach,
            r          = self.r,
        )

        status_colour = "#2ecc71" if result.is_valid else "#e74c3c"
        status_text   = "✓  PASS" if result.is_valid else "✗  FAIL"

        lines = [
            ("MODE",       self.mode.upper().replace("_", " "),  "#aaaaff"),
            ("",           "",                                    "white"),
            ("Nodes (V_int)", str(len(self.nodes)),              "#dddddd"),
            ("Ring size r",   str(self.r),                       "#dddddd"),
            ("V total",       str(len(self.nodes) + self.r),     "#dddddd"),
            ("",           "",                                    "white"),
            ("E_internal", str(len(self.edges)),                 "#dddddd"),
            ("E_attachment",  str(self.e_attach),                "#dddddd"),
            ("E_total",    str(len(self.edges) + self.e_attach), "#dddddd"),
            ("E_expected", str(result.E_expected),               "#dddddd"),
            ("",           "",                                    "white"),
            (status_text,  "",                                    status_colour),
        ]

        y = 0.96
        for label, value, colour in lines:
            if label == "":
                y -= 0.025
                continue
            self.info_ax.text(
                0.05, y, f"{label}:", color="#888888", fontsize=9,
                transform=self.info_ax.transAxes, va="top"
            )
            self.info_ax.text(
                0.95, y, value, color=colour, fontsize=9,
                transform=self.info_ax.transAxes, va="top", ha="right",
                fontweight="bold" if label.startswith(("✓", "✗")) else "normal"
            )
            y -= 0.055

        # Legend
        y -= 0.04
        self.info_ax.text(0.05, y, "Node shapes:", color="#888888",
                          fontsize=8, transform=self.info_ax.transAxes, va="top")
        y -= 0.05
        for shape, deg in DEGREE_MAP.items():
            self.info_ax.text(0.08, y, f"● {shape}  =  deg {deg}",
                              color=COLOURS[shape], fontsize=7.5,
                              transform=self.info_ax.transAxes, va="top")
            y -= 0.048

        y -= 0.03
        hints = [
            "Click node  → cycle shape",
            "Shift+click → delete node",
            "Click edge  → delete edge",
            "Click empty → add node",
            "A / Esc     → add/cancel edge",
            "+ / -       → ring size ±1",
            "] / [       → E_attach  ±1",
            "S           → save JSON",
        ]
        for h in hints:
            self.info_ax.text(0.05, y, h, color="#666688", fontsize=7,
                              transform=self.info_ax.transAxes, va="top")
            y -= 0.042

    # ── Hit testing ──────────────────────────────────────────────────────────

    def _node_at(self, x: float, y: float) -> int | None:
        best, best_d = None, HIT_RADIUS_PX
        for idx, n in enumerate(self.nodes):
            d = np.hypot(x - n.x, y - n.y)
            if d < best_d:
                best, best_d = idx, d
        return best

    def _edge_at(self, x: float, y: float) -> tuple[int, int] | None:
        best, best_d = None, EDGE_HIT_PX
        for i, j in self.edges:
            n1, n2 = self.nodes[i], self.nodes[j]
            # Distance from point to segment
            dx, dy = n2.x - n1.x, n2.y - n1.y
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                continue
            t = max(0.0, min(1.0, ((x - n1.x) * dx + (y - n1.y) * dy) / length_sq))
            px, py = n1.x + t * dx, n1.y + t * dy
            d = np.hypot(x - px, y - py)
            if d < best_d:
                best, best_d = (i, j), d
        return best

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_click(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        shift = event.key == "shift" if event.key else False

        if self.mode == "add_edge":
            idx = self._node_at(x, y)
            if idx is not None:
                node_idx: int = idx
                first = self.edge_first
                if first is None:
                    self.edge_first = node_idx
                elif node_idx != first:
                    e = (min(first, node_idx), max(first, node_idx))
                    if e in self.edges:
                        self.edges.discard(e)
                    else:
                        self.edges.add(e)
                    self.edge_first = None
                    self.mode = "normal"
                self.redraw()
            return

        # Normal mode
        idx = self._node_at(x, y)
        if idx is not None:
            if shift:
                self._delete_node(idx)
            else:
                self._cycle_shape(idx)
            return

        edge = self._edge_at(x, y)
        if edge is not None:
            self.edges.discard(edge)
            self.redraw()
            return

        # Click on empty space → add node
        new_node = Node(x=int(round(x)), y=int(round(y)),
                        radius=10, shape="solid_dot", degree=5)
        self.nodes.append(new_node)
        self.redraw()

    def _on_key(self, event: Any) -> None:
        k = event.key
        if k == "a":
            self._toggle_add_edge()
        elif k == "escape":
            self.mode = "normal"
            self.edge_first = None
            self.redraw()
        elif k == "s":
            self._save()
        elif k in ("+", "="):
            self.r = max(3, self.r + 1)
            self.redraw()
        elif k == "-":
            self.r = max(3, self.r - 1)
            self.redraw()
        elif k == "]":
            self.e_attach += 1
            self.redraw()
        elif k == "[":
            self.e_attach = max(0, self.e_attach - 1)
            self.redraw()
        elif k == "n":
            self._advance(+1)
        elif k == "p":
            self._advance(-1)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _cycle_shape(self, idx: int):
        n = self.nodes[idx]
        next_shape = SHAPE_CYCLE[(SHAPE_CYCLE.index(n.shape) + 1) % len(SHAPE_CYCLE)]
        self.nodes[idx] = Node(x=n.x, y=n.y, radius=n.radius,
                               shape=next_shape, degree=DEGREE_MAP[next_shape])
        self.redraw()

    def _delete_node(self, idx: int):
        self.nodes.pop(idx)
        # Re-index edges: remove any involving idx, shift indices above it
        new_edges = set()
        for i, j in self.edges:
            if i == idx or j == idx:
                continue
            new_edges.add((
                i - (1 if i > idx else 0),
                j - (1 if j > idx else 0),
            ))
        self.edges = new_edges
        if self.edge_first == idx:
            self.edge_first = None
        elif self.edge_first is not None and self.edge_first > idx:
            self.edge_first -= 1
        self.redraw()

    def _toggle_add_edge(self):
        if self.mode == "add_edge":
            self.mode = "normal"
            self.edge_first = None
        else:
            self.mode = "add_edge"
            self.edge_first = None
        self.redraw()

    def _save(self):
        result = _validator.check(
            V            = len(self.nodes) + self.r,
            E_internal   = len(self.edges),
            E_attachment = self.e_attach,
            r            = self.r,
        )
        if not result.is_valid:
            print(f"[HITL] Cannot save — validator FAIL: {result}")
            # Flash the info panel red briefly
            self.info_ax.set_facecolor("#3a1a1a")
            self.fig.canvas.draw_idle()
            import threading
            def _reset():
                import time; time.sleep(0.4)
                self.info_ax.set_facecolor("#2a2a3e")
                self.fig.canvas.draw_idle()
            threading.Thread(target=_reset, daemon=True).start()
            return

        stem = self.image_path.stem
        out_path = self.out_dir / f"{stem}.json"

        payload = {
            "source_crop": str(self.image_path),
            "ring_size":   self.r,
            "V_interior":  len(self.nodes),
            "V_total":     len(self.nodes) + self.r,
            "E_internal":  len(self.edges),
            "E_attachment": self.e_attach,
            "euler_valid": True,
            "nodes": [
                {"id": i, "shape": n.shape, "degree": n.degree,
                 "x": n.x, "y": n.y}
                for i, n in enumerate(self.nodes)
            ],
            "edges": [
                {"from": i, "to": j} for i, j in sorted(self.edges)
            ],
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[HITL] Saved → {out_path}")
        self.info_ax.set_facecolor("#1a3a1a")
        self.fig.canvas.draw_idle()
        import threading
        def _reset():
            import time; time.sleep(0.6)
            self.info_ax.set_facecolor("#2a2a3e")
            self.fig.canvas.draw_idle()
        threading.Thread(target=_reset, daemon=True).start()

    def _advance(self, direction: int):
        # Signal to the outer loop which image to open next
        self._next_direction = direction
        plt.close(self.fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HITL correction UI for Appel-Haken configurations")
    parser.add_argument("path", help="Path to a single crop PNG or a directory of PNGs")
    parser.add_argument("--out", default="annotations", help="Output directory for JSON files")
    parser.add_argument("--detections", type=Path, default=None,
                        help="detections_part2.json from batch_detect.py — pre-populates nodes")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Skip crops with zero detected nodes (only with --detections)")
    args = parser.parse_args()

    # Load pre-computed detections if provided
    detections: dict[str, list[Node]] = {}
    detection_order: list[str] = []   # crop filenames in JSON order
    if args.detections:
        raw = json.loads(args.detections.read_text())
        for entry in raw:
            if args.skip_empty and entry["node_count"] == 0:
                continue
            nodes = [Node(**n) for n in entry["nodes"]]
            detections[entry["crop"]] = nodes
            detection_order.append(entry["crop"])

    p = Path(args.path)
    if p.is_dir():
        if detection_order:
            # Use JSON order; only include crops that exist on disk
            images = [p / name for name in detection_order if (p / name).exists()]
        else:
            images = sorted(p.glob("*_cell*.png"))
            images = [i for i in images if "_nodes" not in i.name and "_debug" not in i.name]
    elif p.is_file():
        images = [p]
    else:
        sys.exit(f"Path not found: {p}")

    if not images:
        sys.exit("No crop images found.")

    idx = 0
    while 0 <= idx < len(images):
        preloaded = detections.get(images[idx].name) if detections else None
        editor = HITLEditor(str(images[idx]), out_dir=args.out, preloaded_nodes=preloaded)
        direction = getattr(editor, "_next_direction", 0)
        idx += direction if direction != 0 else len(images)  # 0 = window closed normally → stop


if __name__ == "__main__":
    main()
