"""The real scene, as geometry (src/simulation/render/scene_mesh.py).

Turns a depth map and the camera frame it came from into a triangle mesh in LAB
WORLD coordinates, coloured from the image. It is what lets the reenactment be
staged in the user's actual room with everything in the shot being real
geometry - lit by the same rig, depth-tested against the hand and the object -
rather than 3-D actors composited over a flat video plate.

FRAME. Depth is in the PERCEPTION frame (+X right, +Y DOWN, +Z away, metres)
and is back-projected with the pinhole the rest of the system assumes,
fx = fy = 0.8 * width with the principal point at the image centre. The result
is flipped into lab world (+Y up, +Z toward the viewer) on the way out, so it
lands in the same space as the object and hand meshes.

DISCONTINUITIES. A depth map is a surface seen from one side, so neighbouring
pixels that straddle an object's edge are metres apart in Z while adjacent on
screen. Triangulating those naively drapes a rubber sheet from the foreground to
the back wall - the single most conspicuous artefact of this kind of
reconstruction. Cells spanning more than a proportional depth jump are dropped
instead, which leaves an honest hole where the sensor genuinely saw nothing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.simulation.render.raster import Material, Mesh

# Perception -> lab world, the same flip lab_sim uses.
_AXIS_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

# A cell is dropped when its depth range exceeds this fraction of its own mean
# depth. Proportional rather than absolute because a 5 cm step matters at 30 cm
# and is nothing at 3 m.
_MAX_RELATIVE_DEPTH_JUMP = 0.06

# The scene is background: it should read as a room, not compete with the
# actors, and every triangle is one the rasteriser pays for on every frame.
_SCENE_MATERIAL = Material(specular=0.04, shininess=8.0, cull_backfaces=False)


def build_scene_mesh(
    depth_m: np.ndarray,
    frame_bgr: np.ndarray,
    grid_w: int = 72,
    max_relative_jump: float = _MAX_RELATIVE_DEPTH_JUMP,
    min_depth_m: float = 0.05,
    max_depth_m: float = 8.0,
) -> Optional[Mesh]:
    """Reconstruct `depth_m` as a coloured mesh, or None if there is nothing to build.

    `depth_m` is metres, any resolution; `frame_bgr` is the image it was
    estimated from. Both are resampled onto a `grid_w`-wide lattice at the
    frame's aspect - the mesh is background, and a vertex per pixel would cost
    far more than it shows.
    """
    if depth_m is None or frame_bgr is None or depth_m.size == 0:
        return None
    fh, fw = frame_bgr.shape[:2]
    if fw < 2 or fh < 2:
        return None

    gw = int(max(8, min(int(grid_w), fw)))
    gh = int(max(6, round(gw * fh / float(fw))))

    # NEAREST for depth: averaging across a silhouette invents a surface that
    # is at neither of the depths it sits between, which is exactly the edge the
    # jump test below is trying to find.
    z = cv2.resize(depth_m.astype(np.float32), (gw, gh),
                   interpolation=cv2.INTER_NEAREST)
    rgb = cv2.resize(frame_bgr, (gw, gh), interpolation=cv2.INTER_AREA)

    valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    if not valid.any():
        return None
    z = np.where(valid, z, np.nan).astype(np.float32)

    # Sample positions in ORIGINAL pixel coordinates, so the reconstruction
    # projects back onto the pixels it was built from.
    us = (np.arange(gw, dtype=np.float32) + 0.5) * (fw / float(gw))
    vs = (np.arange(gh, dtype=np.float32) + 0.5) * (fh / float(gh))
    uu, vv = np.meshgrid(us, vs)

    fx = 0.8 * float(fw)
    cx, cy = fw * 0.5, fh * 0.5
    zz = np.nan_to_num(z, nan=1.0)
    x = (uu - cx) * zz / fx
    y = (vv - cy) * zz / fx
    pts_cam = np.stack([x, y, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    vertices = (pts_cam @ _AXIS_FLIP.T).astype(np.float32)
    colors = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)

    # Two triangles per cell, dropped where the cell straddles a depth edge or
    # touches an invalid sample.
    i0 = np.arange(gh - 1)[:, None] * gw + np.arange(gw - 1)[None, :]
    a = i0.ravel()
    b = a + 1
    c = a + gw
    d = c + 1

    quad = np.stack([zz.ravel()[a], zz.ravel()[b], zz.ravel()[c], zz.ravel()[d]], 1)
    ok = np.stack([valid.ravel()[a], valid.ravel()[b],
                   valid.ravel()[c], valid.ravel()[d]], 1).all(axis=1)
    spread = quad.max(axis=1) - quad.min(axis=1)
    ok &= spread <= max_relative_jump * np.maximum(quad.mean(axis=1), 1e-3)
    if not ok.any():
        return None

    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)],
                           axis=0).astype(np.int32)

    normals = _vertex_normals(vertices, faces)
    return Mesh(vertices=vertices, faces=faces, normals=normals,
                colors=colors.astype(np.float32), material=_SCENE_MATERIAL)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, which is what accumulating the raw face
    cross-products gives: a bigger triangle contributes proportionally more."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # An unreferenced vertex (its cell was dropped) has no normal to speak of.
    # Facing it at the camera is harmless and keeps the array unit-length.
    normals = np.where(lengths > 1e-9, normals / np.maximum(lengths, 1e-9),
                       np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return normals.astype(np.float32)
