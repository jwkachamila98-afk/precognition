"""Build a solid hand mesh from the 21-joint keypoint skeleton.

The joint layout is the MANO/MediaPipe convention used everywhere in this
project: 0 = wrist, then thumb, index, middle, ring, pinky in blocks of four
(MCP, PIP, DIP, TIP).

The palm is an extruded outline rather than a bundle of capsules - a real palm
is a flat shell, and extruding its outline is what gives the silhouette its
recognisable shape. Fingers are swept tubes with rotation-minimising frames, so
they bend without twisting.

This is a *procedural* hand built from tracked joint positions. It is not the
MANO parametric surface model itself: MANO's learned shape/pose blend weights
require a licence from the MPI project site, so they cannot be shipped or
fetched here. Proportions below are anthropometric ratios of the measured palm
length, which is why the mesh looks right for whatever hand the tracker sees.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.simulation.render import primitives as P
from src.simulation.render.raster import Material, Mesh, concat_meshes, orient_faces_outward

WRIST = 0
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_MCP = {name: chain[0] for name, chain in FINGER_CHAINS.items()}

# Radii as a fraction of measured palm length (wrist -> middle MCP, ~9 cm on an
# adult hand). Base at the knuckle, tip at the fingertip.
_FINGER_RADII = {
    "thumb": (0.138, 0.100),
    "index": (0.108, 0.076),
    "middle": (0.112, 0.078),
    "ring": (0.104, 0.072),
    "pinky": (0.092, 0.064),
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _catmull_rom(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline through a centripetal Catmull-Rom spline.

    Straight-line finger segments read as faceted; a spline through the four
    tracked joints gives the continuous curvature a real finger has.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3 or count <= len(pts):
        return pts
    ext = np.concatenate([pts[:1] * 2 - pts[1:2], pts, pts[-1:] * 2 - pts[-2:-1]])
    n_seg = len(pts) - 1
    ts = np.linspace(0.0, n_seg, count, dtype=np.float32)
    out = np.empty((count, 3), dtype=np.float32)
    for k, t in enumerate(ts):
        i = min(int(t), n_seg - 1)
        u = t - i
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        u2, u3 = u * u, u * u * u
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * u
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
    return out


def palm_length(kpts: np.ndarray) -> float:
    """Wrist -> middle-MCP distance: the scale reference for every proportion."""
    return max(float(np.linalg.norm(kpts[_MCP["middle"]] - kpts[WRIST])), 1e-5)


def palm_normal(kpts: np.ndarray) -> np.ndarray:
    """Unit normal of the palm plane, from the knuckle span and the wrist axis."""
    across = kpts[_MCP["pinky"]] - kpts[_MCP["index"]]
    along = kpts[_MCP["middle"]] - kpts[WRIST]
    n = np.cross(across, along)
    if float(np.linalg.norm(n)) < 1e-8:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _unit(n)


def _palm_slab(kpts: np.ndarray, scale: float, color, material: Material) -> Mesh:
    """Extrude the palm outline into a slab with a hard rim."""
    n = palm_normal(kpts)
    across = _unit(kpts[_MCP["pinky"]] - kpts[_MCP["index"]])
    wrist_half = 0.34 * float(np.linalg.norm(kpts[_MCP["pinky"]] - kpts[_MCP["index"]]))

    loop = np.stack([
        kpts[WRIST] - across * wrist_half,
        kpts[FINGER_CHAINS["thumb"][0]],
        kpts[_MCP["index"]],
        kpts[_MCP["middle"]],
        kpts[_MCP["ring"]],
        kpts[_MCP["pinky"]],
        kpts[WRIST] + across * wrist_half,
    ]).astype(np.float32)

    # Inflate outward from the centroid: tracked joints sit inside the flesh.
    centroid = loop.mean(axis=0)
    radial = loop - centroid
    radial -= n[None, :] * (radial @ n)[:, None]
    lens = np.linalg.norm(radial, axis=1, keepdims=True)
    loop = loop + (radial / np.maximum(lens, 1e-8)) * (0.115 * scale)

    thickness = 0.30 * scale
    half = n * (thickness * 0.5)
    front, back = loop + half, loop - half
    k = len(loop)

    # The cap centroids are pushed out along the palm normal, doming both faces.
    # A flat prism reads as a playing card from any angle where the fingers are
    # foreshortened; the dome costs two moved vertices and reads as a palm.
    dome_front = front.mean(axis=0) + n * (thickness * 0.40)
    dome_back = back.mean(axis=0) - n * (thickness * 0.22)
    verts = [front, back, np.array([dome_front, dome_back], dtype=np.float32)]
    norms = [np.tile(n, (k, 1)), np.tile(-n, (k, 1)), np.array([n, -n], dtype=np.float32)]
    faces = []
    cf, cb = 2 * k, 2 * k + 1
    for i in range(k):
        j = (i + 1) % k
        faces.append([cf, i, j])
        faces.append([cb, k + j, k + i])

    off = 2 * k + 2
    side_v, side_n = [], []
    for i in range(k):
        j = (i + 1) % k
        sn = _unit(np.cross(loop[j] - loop[i], n))
        base = off + 4 * i
        side_v.extend([front[i], front[j], back[j], back[i]])
        side_n.extend([sn, sn, sn, sn])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    verts.append(np.array(side_v, dtype=np.float32))
    norms.append(np.array(side_n, dtype=np.float32))

    v = np.concatenate(verts).astype(np.float32)
    nn = np.concatenate(norms).astype(np.float32)
    col = np.tile(np.asarray(color, dtype=np.float32).reshape(3), (len(v), 1))
    return Mesh(v, np.array(faces, dtype=np.int32), nn, col, material=material)


def build_hand_mesh(
    keypoints_3d: np.ndarray,
    color=(0.85, 0.72, 0.35),
    material: Optional[Material] = None,
    radial_segments: int = 6,
    spline_points: int = 5,
) -> Mesh:
    """Solid hand mesh from (21, 3) keypoints already in LAB WORLD coordinates.

    ~390 triangles at the defaults, merged into a single draw batch.
    """
    kpts = np.asarray(keypoints_3d, dtype=np.float32).reshape(21, 3)
    scale = palm_length(kpts)
    mat = material or Material(specular=0.25, shininess=40.0)

    # Each part orients its own winding (primitives do this internally; the palm
    # slab is oriented here). It must happen PER PART, never on the merged hand:
    # the outward test compares a face against its part's centroid, and run on
    # the whole hand it would test fingertip faces against the palm centroid and
    # flip them, punching holes in the fingers once culling is on.
    parts = [orient_faces_outward(_palm_slab(kpts, scale, color, mat))]

    for name, chain in FINGER_CHAINS.items():
        joints = kpts[list(chain)]
        # Start the tube inside the palm slab so no seam opens at the knuckle.
        root = joints[0] + (joints[0] - joints[1]) * 0.32
        path = _catmull_rom(np.concatenate([root[None, :], joints]), spline_points + 1)
        r_base, r_tip = _FINGER_RADII[name]
        radii = np.linspace(r_base * scale, r_tip * scale, len(path), dtype=np.float32)
        parts.append(P.tube(path, radii, color, segments=radial_segments,
                            cap_start=False, cap_end=True, material=mat))

    return concat_meshes(parts)
