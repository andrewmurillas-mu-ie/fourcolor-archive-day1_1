"""Canonical data model for Appel–Haken configurations (schema v1.1).

Merged 2026-08-14 from the fourcolor-archive day-1 repo into the main
pipeline. A *configuration* is the interior of a triangulated disk: interior
vertices with Heesch degree specifications plus internal edges. The bounding
ring is NOT drawn in the source material — its size is implied (validator.py).

Two representations coexist:
  * pipeline detection dicts (batch_detect.py / detections_part.json)
  * this canonical model, used for the published dataset

`Configuration.from_detection` bridges them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = "1.1"

#: Heesch symbol -> specified vertex degree.
SHAPE_DEGREES = {
    "solid_dot": 5,
    "open_circle": 6,
    "square": 7,
    "triangle": 8,
}
DEGREE_SHAPES = {d: s for s, d in SHAPE_DEGREES.items()}


@dataclass
class Vertex:
    id: int
    degree: int                     # degree in the FULL triangulation
    shape: Optional[str] = None     # Heesch symbol as drawn
    pos: Optional[tuple[float, float]] = None  # pixel coords in source crop
    radius: Optional[int] = None    # detected marker radius (px), for HITL

    def __post_init__(self) -> None:
        if self.shape in SHAPE_DEGREES and self.shape != "triangle":
            expected = SHAPE_DEGREES[self.shape]
            if self.degree != expected:
                raise ValueError(
                    f"vertex {self.id}: shape {self.shape!r} implies degree "
                    f"{expected}, got {self.degree}"
                )
        elif self.shape == "triangle" and self.degree < 8:
            raise ValueError(
                f"vertex {self.id}: triangle means degree >= 8, "
                f"got {self.degree}"
            )


@dataclass
class Provenance:
    source_document: Optional[str] = None   # e.g. "Part II PDF, p. 14"
    crop: Optional[str] = None              # e.g. "page014_cell000.png"
    figure_ref: Optional[str] = None        # e.g. "CTL #101" once OCR'd
    extraction: str = "manual"              # "manual" | "cv" | "cv+hitl"
    human_corrected: bool = False
    notes: Optional[str] = None


@dataclass
class Configuration:
    id: str
    vertices: list[Vertex]
    edges: list[tuple[int, int]]            # internal edges only
    ring_size: Optional[int] = None         # labeled/expected ring size
    reducibility: Optional[str] = None      # "D" | "C" | None
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_internal_edges(self) -> int:
        return len(self.edges)

    def degree_sum(self) -> int:
        return sum(v.degree for v in self.vertices)

    def internal_degree(self) -> dict[int, int]:
        d = {v.id: 0 for v in self.vertices}
        for u, w in self.edges:
            d[u] += 1
            d[w] += 1
        return d

    # ------------------------------------------------------------------ #
    # Bridges to/from the pipeline's detection dict format
    # ------------------------------------------------------------------ #
    @classmethod
    def from_detection(cls, det: dict,
                       labeled_ring: Optional[int] = None,
                       reducibility: Optional[str] = None) -> "Configuration":
        """Build a Configuration from a batch_detect/HITL detection dict.

        `labeled_ring` is EXTERNAL ground truth (e.g. OCR'd caption or the
        ring-size section of Part II the crop came from). It is deliberately
        NOT taken from det["ring_size"], which is derived from the detected
        topology and therefore cannot be used to check that same topology.
        """
        vertices = [
            Vertex(
                id=i,
                degree=n["degree"],
                shape=n.get("shape"),
                pos=(n["x"], n["y"]) if "x" in n else None,
                radius=n.get("radius"),
            )
            for i, n in enumerate(det.get("nodes", []))
        ]
        edges = [(e["from"], e["to"]) for e in det.get("edges", [])]
        return cls(
            id=det.get("crop", "unknown").replace(".png", ""),
            vertices=vertices,
            edges=edges,
            ring_size=labeled_ring,
            reducibility=reducibility,
            provenance=Provenance(
                source_document=f"Part II PDF, page {det.get('page')}",
                crop=det.get("crop"),
                extraction="cv+hitl" if det.get("human_corrected") else "cv",
                human_corrected=bool(det.get("human_corrected")),
            ),
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        out["edges"] = [list(e) for e in self.edges]
        return out

    def to_json(self, **kwargs) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Configuration":
        d = dict(d)
        d.pop("schema_version", None)
        d["vertices"] = [
            Vertex(
                id=v["id"], degree=v["degree"], shape=v.get("shape"),
                pos=tuple(v["pos"]) if v.get("pos") else None,
                radius=v.get("radius"),
            )
            for v in d["vertices"]
        ]
        d["edges"] = [tuple(e) for e in d["edges"]]
        d["provenance"] = Provenance(**(d.get("provenance") or {}))
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "Configuration":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------- #
# Hand-coded ground truth (regression anchors)
# ---------------------------------------------------------------------- #
def birkhoff_diamond() -> Configuration:
    """Birkhoff's diamond: 4 interior 5-vertices, E_int=5, ring 6.

    NOTE: this configuration appears in our own scans as
    data/crops/page014_cell000.png — where the June batch run detected only
    4 of its 5 edges and still auto-passed with ring 7. That false positive
    is the motivating example for labeled-ring cross-checking.
    """
    diamond_pos = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    return Configuration(
        id="birkhoff-diamond",
        vertices=[Vertex(i, 5, "solid_dot", pos=diamond_pos[i])
                  for i in range(4)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        ring_size=6,
        reducibility="D",
        provenance=Provenance(
            source_document="Birkhoff 1913; also page014_cell000.png",
            extraction="manual",
            notes="Hand-coded ground truth (Casey Phase 3 milestone).",
        ),
    )


def single_five_vertex() -> Configuration:
    """A lone degree-5 vertex: trivial configuration, ring size 5."""
    return Configuration(
        id="single-v5",
        vertices=[Vertex(0, 5, "solid_dot")],
        edges=[],
        ring_size=5,
        provenance=Provenance(extraction="manual",
                              notes="Hand-coded ground truth."),
    )
