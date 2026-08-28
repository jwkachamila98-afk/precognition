"""Static geometry for the simulated robotics lab.

Authored in LAB WORLD metres (+X right, +Y up, +Z toward the viewer) with the
floor at y = 0 and the workbench top at ``BENCH_TOP_Y``. Built once per session
and rasterized into a cached G-buffer, so triangle count here costs a one-off,
not a per-frame, price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.simulation.render import primitives as P
from src.simulation.render import textures as T
from src.simulation.render.raster import Material, Mesh

BENCH_TOP_Y = 0.92
BENCH_HALF_X = 0.90
BENCH_FRONT_Z = 0.42
BENCH_BACK_Z = -0.46
PEDESTAL_TOP_Y = BENCH_TOP_Y + 0.014
OBJECT_ANCHOR = np.array([0.0, PEDESTAL_TOP_Y, 0.02], dtype=np.float32)

_ROOM_HALF_X = 2.6
_ROOM_BACK_Z = -2.3
_ROOM_FRONT_Z = 2.6
_ROOM_TOP_Y = 2.75

_MATTE = Material(specular=0.05, shininess=12.0)
_SEMI_GLOSS = Material(specular=0.30, shininess=54.0)
_METAL = Material(specular=0.55, shininess=90.0)
_EMISSIVE = Material(specular=0.0, shininess=1.0, emissive=1.10, cull_backfaces=False)
_RING_LIGHT = Material(specular=0.0, shininess=1.0, emissive=1.6, cull_backfaces=False)
_FLOOR_MAT = Material(specular=0.22, shininess=40.0, cull_backfaces=False, bilinear=True)
_PANEL_MAT = Material(specular=0.08, shininess=16.0, cull_backfaces=False, bilinear=True)
_TOP_MAT = Material(specular=0.26, shininess=62.0, bilinear=True)


@dataclass
class LabScene:
    """The static lab: meshes plus the anchors the dynamic actors need."""

    meshes: List[Mesh]
    bench_top_y: float = BENCH_TOP_Y
    object_anchor: np.ndarray = None

    def __post_init__(self) -> None:
        if self.object_anchor is None:
            self.object_anchor = OBJECT_ANCHOR.copy()

    @property
    def triangle_count(self) -> int:
        return sum(m.num_faces for m in self.meshes)


def _floor() -> Mesh:
    y = 0.0
    return P.quad(
        (-_ROOM_HALF_X, y, _ROOM_FRONT_Z),
        (_ROOM_HALF_X, y, _ROOM_FRONT_Z),
        (_ROOM_HALF_X, y, _ROOM_BACK_Z),
        (-_ROOM_HALF_X, y, _ROOM_BACK_Z),
        color=(1.0, 1.0, 1.0),
        uvs=[[0, 1], [3.2, 1], [3.2, 0], [0, 0]],
        texture=T.epoxy_floor(),
        material=_FLOOR_MAT,
    )


def _walls() -> List[Mesh]:
    tex = T.wall_panel()
    back = P.quad(
        (-_ROOM_HALF_X, 0.0, _ROOM_BACK_Z),
        (_ROOM_HALF_X, 0.0, _ROOM_BACK_Z),
        (_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        (-_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        color=(1.0, 1.0, 1.0), uvs=[[0, 1], [3, 1], [3, 0], [0, 0]],
        texture=tex, material=_PANEL_MAT,
    )
    left = P.quad(
        (-_ROOM_HALF_X, 0.0, _ROOM_BACK_Z),
        (-_ROOM_HALF_X, 0.0, _ROOM_FRONT_Z),
        (-_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_FRONT_Z),
        (-_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        color=(0.88, 0.88, 0.88), uvs=[[0, 1], [3, 1], [3, 0], [0, 0]],
        texture=tex, material=_PANEL_MAT,
    )
    right = P.quad(
        (_ROOM_HALF_X, 0.0, _ROOM_FRONT_Z),
        (_ROOM_HALF_X, 0.0, _ROOM_BACK_Z),
        (_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        (_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_FRONT_Z),
        color=(0.88, 0.88, 0.88), uvs=[[0, 1], [3, 1], [3, 0], [0, 0]],
        texture=tex, material=_PANEL_MAT,
    )
    ceiling = P.quad(
        (-_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        (_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_BACK_Z),
        (_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_FRONT_Z),
        (-_ROOM_HALF_X, _ROOM_TOP_Y, _ROOM_FRONT_Z),
        color=(0.20, 0.20, 0.21), material=Material(specular=0.02, shininess=8.0,
                                                    cull_backfaces=False),
    )
    return [back, left, right, ceiling]


def _workbench() -> List[Mesh]:
    top_thick = 0.055
    depth = BENCH_FRONT_Z - BENCH_BACK_Z
    cz = (BENCH_FRONT_Z + BENCH_BACK_Z) / 2.0
    parts = [
        P.box((0.0, BENCH_TOP_Y - top_thick / 2.0, cz),
              (2 * BENCH_HALF_X, top_thick, depth),
              (0.74, 0.73, 0.71), material=_TOP_MAT, texture=T.brushed_steel(), uv_scale=2.0),
        # Dark apron under the worktop edge - gives the slab visible thickness.
        P.box((0.0, BENCH_TOP_Y - top_thick - 0.035, cz),
              (2 * BENCH_HALF_X - 0.03, 0.07, depth - 0.03),
              (0.16, 0.15, 0.14), material=_MATTE),
    ]
    leg_h = BENCH_TOP_Y - top_thick - 0.075
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(P.box(
                (sx * (BENCH_HALF_X - 0.09), leg_h / 2.0, cz + sz * (depth / 2.0 - 0.09)),
                (0.055, leg_h, 0.055), (0.34, 0.33, 0.31), material=_METAL))
    # Lower shelf with a couple of stowed cases.
    parts.append(P.box((0.0, 0.26, cz), (2 * BENCH_HALF_X - 0.26, 0.03, depth - 0.20),
                       (0.26, 0.25, 0.24), material=_MATTE))
    parts.append(P.box((-0.42, 0.35, cz - 0.02), (0.36, 0.15, 0.30),
                       (0.15, 0.16, 0.18), material=_SEMI_GLOSS))
    parts.append(P.box((0.30, 0.33, cz + 0.01), (0.30, 0.11, 0.26),
                       (0.20, 0.19, 0.18), material=_SEMI_GLOSS))
    return parts


def _staging_pedestal() -> List[Mesh]:
    """The lit turntable the manipuland is staged on."""
    y = BENCH_TOP_Y
    return [
        # Light, semi-gloss stage plate: a dark plate swallows the contact
        # shadow that anchors the object to it.
        P.cylinder((0.0, y + 0.007, OBJECT_ANCHOR[2]), 0.135, 0.014,
                   (0.40, 0.395, 0.385), segments=28, material=_SEMI_GLOSS),
        # Inlaid amber edge light. Amber, not cyan: cyan is the hologram's
        # colour, and the staging ring must not compete with the actor.
        P.ring((0.0, y + 0.0146, OBJECT_ANCHOR[2]), 0.136, 0.150,
               (0.16, 0.58, 0.98), segments=32, material=_RING_LIGHT),
    ]


def _backdrop() -> List[Mesh]:
    """A shadow-box panel behind the staging area, so the object reads clean."""
    z = BENCH_BACK_Z - 0.02
    return [
        P.quad((-0.95, BENCH_TOP_Y, z), (0.95, BENCH_TOP_Y, z),
               (0.95, BENCH_TOP_Y + 0.98, z), (-0.95, BENCH_TOP_Y + 0.98, z),
               color=(0.30, 0.29, 0.285), texture=T.backdrop_panel(),
               uvs=[[0, 1], [1.6, 1], [1.6, 0], [0, 0]],
               material=Material(specular=0.05, shininess=12.0,
                                 cull_backfaces=False, bilinear=True)),
        P.box((0.0, BENCH_TOP_Y + 0.99, z), (1.94, 0.035, 0.05),
              (0.30, 0.29, 0.28), material=_METAL),
        P.box((0.0, BENCH_TOP_Y + 0.005, z), (1.94, 0.03, 0.05),
              (0.30, 0.29, 0.28), material=_METAL),
    ]


def _instruments() -> List[Mesh]:
    """Props that make the room legible as a vision/manipulation lab."""
    z = BENCH_BACK_Z - 0.015
    parts = [
        # Checkerboard calibration target, mounted on the backdrop.
        P.quad((-0.63, BENCH_TOP_Y + 0.27, z), (-0.32, BENCH_TOP_Y + 0.27, z),
               (-0.32, BENCH_TOP_Y + 0.58, z), (-0.63, BENCH_TOP_Y + 0.58, z),
               color=(1.0, 1.0, 1.0), texture=T.calibration_target(),
               material=Material(specular=0.10, shininess=20.0, cull_backfaces=False,
                                 bilinear=True)),
        # Telemetry monitor on the opposite side.
        P.quad((0.30, BENCH_TOP_Y + 0.28, z), (0.72, BENCH_TOP_Y + 0.28, z),
               (0.72, BENCH_TOP_Y + 0.55, z), (0.30, BENCH_TOP_Y + 0.55, z),
               color=(1.0, 1.0, 1.0), texture=T.monitor_screen(),
               material=Material(specular=0.20, shininess=60.0, emissive=0.55,
                                 cull_backfaces=False, bilinear=True)),
        # Equipment rack standing beside the bench.
        P.box((-1.42, 0.62, BENCH_BACK_Z + 0.20), (0.52, 1.24, 0.46),
              (1.0, 1.0, 1.0), material=Material(specular=0.18, shininess=30.0,
                                                 bilinear=True),
              texture=T.rack_face()),
        # Overhead truss carrying the light bars.
        P.box((0.0, 2.34, -0.30), (4.0, 0.07, 0.07), (0.30, 0.29, 0.28), material=_METAL),
        P.box((0.0, 2.34, 0.55), (4.0, 0.07, 0.07), (0.30, 0.29, 0.28), material=_METAL),
    ]
    # Two softbox key panels, emissive so they read as the light source on camera.
    for pz in (-0.30, 0.55):
        parts.append(P.box((0.0, 2.28, pz), (1.7, 0.05, 0.26),
                           (0.98, 0.96, 0.92), material=_EMISSIVE))
    # Camera mast: the rig 'observing' the cell.
    parts.append(P.cylinder((0.86, 1.55, BENCH_BACK_Z + 0.06), 0.022, 1.26,
                            (0.28, 0.28, 0.29), segments=10, material=_METAL))
    parts.append(P.box((0.86, 2.20, BENCH_BACK_Z + 0.14), (0.10, 0.08, 0.16),
                       (0.12, 0.12, 0.13), material=_SEMI_GLOSS))
    parts.append(P.cylinder((0.86, 2.20, BENCH_BACK_Z + 0.23), 0.028, 0.05,
                            (0.05, 0.05, 0.06), segments=12, axis="z", material=_METAL))
    return parts


def build_lab() -> LabScene:
    """Assemble the full static lab."""
    meshes: List[Mesh] = [_floor()]
    meshes += _walls()
    meshes += _workbench()
    meshes += _backdrop()
    meshes += _staging_pedestal()
    meshes += _instruments()
    return LabScene(meshes=meshes)
