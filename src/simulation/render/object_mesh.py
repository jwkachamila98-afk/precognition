"""Reconstruct a textured 3-D mesh of the real target object from one RGB crop.

What is actually real here, and what is inferred - stated plainly, because the
difference matters:

  * REAL: the object's silhouette and its surface colour. Both come from
    ``LocalVisualizer.capture_object_sprite``, a genuine photo-crop of the
    object taken from the live camera while it was unoccluded. The silhouette is
    segmented with GrabCut; the crop is used directly as the albedo texture, so
    the object in the lab wears its own photograph.
  * INFERRED: depth. A single RGB view carries no depth, and this machine has no
    depth sensor (``MockDepthEstimator`` is what runs locally - see
    local_client.py). So the silhouette is *inflated* into a shell whose
    thickness follows the distance transform of the mask - the standard
    silhouette-inflation used by sketch-based modellers (Teddy/FiberMesh). Thick
    where the object is wide, thin where it tapers.

Consequence worth knowing: inflation assumes a roughly star-shaped silhouette
about the mask centroid. A remote, a bottle, or a box reconstruct faithfully; a
mug handle or another deep concavity will be smoothed over rather than hollowed.
That is the honest ceiling of monocular single-view reconstruction, not a bug in
the implementation.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.simulation.render.raster import Material, Mesh, orient_faces_outward

_OBJECT_MATERIAL = Material(specular=0.26, shininess=44.0, bilinear=True)


def segment_object(sprite: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Foreground mask for the object in its crop, as uint8 {0, 255}.

    GrabCut seeded with an inset rectangle; falls back to an inscribed ellipse
    if it fails or returns a degenerate mask (which happens on low-contrast
    crops).
    """
    h, w = sprite.shape[:2]
    fallback = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(fallback, (w // 2, h // 2), (max(1, int(w * 0.46)), max(1, int(h * 0.46))),
                0, 0, 360, 255, -1)
    if h < 16 or w < 16:
        return fallback

    try:
        mask = np.zeros((h, w), dtype=np.uint8)
        inset_x = max(1, int(w * 0.06))
        inset_y = max(1, int(h * 0.06))
        rect = (inset_x, inset_y, w - 2 * inset_x, h - 2 * inset_y)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(sprite, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return fallback

    if fg.sum() < 0.04 * 255 * h * w:
        return fallback

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n_labels <= 1:
        return fallback
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    return fg if fg.sum() > 0.03 * 255 * h * w else fallback


def _resample_contour(contour: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed contour to `count` points evenly spaced by arc length."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], count, axis=0)
    targets = np.linspace(0.0, total, count, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    frac = ((targets - cum[idx]) / np.maximum(seg[idx], 1e-6))[:, None]
    return closed[idx] + (closed[idx + 1] - closed[idx]) * frac


def build_object_mesh(
    sprite: np.ndarray,
    longest_dim_m: float,
    depth_ratio: float = 0.45,
    contour_points: int = 26,
    rings: int = 3,
    back_depth_ratio: float = 0.62,
) -> Optional[Mesh]:
    """Silhouette-inflated, photo-textured mesh centred on the origin.

    The caller supplies only SCALE (the object's longest real dimension); the
    reconstruction supplies SHAPE, taken from the segmented silhouette's own
    aspect ratio. Splitting it this way matters: the detector's 3-D extent comes
    from synthetic depth locally and routinely disagrees with what the object
    actually looks like, and trusting it for aspect turns a wine glass into a
    lozenge. The silhouette is a direct measurement of the projected shape.

    Local frame: +X right, +Y up, +Z toward the viewer; the object is centred on
    x/z with its base at ``y = -height / 2``.
    """
    if sprite is None or sprite.size == 0 or min(sprite.shape[:2]) < 8:
        return None

    mask = segment_object(sprite)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 24:
        return None

    h, w = mask.shape[:2]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_max = float(dist.max())
    if dist_max < 1e-3:
        return None

    m = cv2.moments(mask, binaryImage=True)
    cx = float(m["m10"] / max(m["m00"], 1e-6))
    cy = float(m["m01"] / max(m["m00"], 1e-6))

    rim = _resample_contour(contour, contour_points)          # (N, 2) pixels
    center = np.array([cx, cy], dtype=np.float32)

    # Radial fan: ring 0 is the centroid, ring `rings` is the true silhouette.
    fractions = np.linspace(0.0, 1.0, rings + 1, dtype=np.float32)
    ring_px = [center[None, :] + f * (rim - center[None, :]) for f in fractions]

    # Pixel -> metres. The mask bbox spans the object's real extent, so scale by
    # it rather than by the crop, which carries background margin.
    bx, by, bw, bh = cv2.boundingRect(contour)
    longest = max(float(longest_dim_m), 0.01)
    if bw >= bh:
        sx, sy = longest, longest * bh / max(bw, 1)
    else:
        sy, sx = longest, longest * bw / max(bh, 1)
    sz = float(np.clip(depth_ratio * min(sx, sy), 0.008, 0.9 * min(sx, sy)))
    px_to_m_x = sx / max(bw, 1)
    px_to_m_y = sy / max(bh, 1)
    depth_amp = sz * 0.5

    verts, norms, uvs = [], [], []
    for r, ring in enumerate(ring_px):
        px = np.clip(ring[:, 0], 0, w - 1)
        py = np.clip(ring[:, 1], 0, h - 1)
        d = dist[np.rint(py).astype(np.int32), np.rint(px).astype(np.int32)] / dist_max
        z_front = depth_amp * np.sqrt(np.clip(d, 0.0, 1.0))
        if r == 0:
            # Collapse ring 0 to a single apex vertex.
            px = px[:1] * 0 + cx
            py = py[:1] * 0 + cy
            z_front = np.array([depth_amp], dtype=np.float32)
        wx = (px - (bx + bw * 0.5)) * px_to_m_x
        wy = -(py - (by + bh * 0.5)) * px_to_m_y
        uv = np.stack([px / max(w - 1, 1), py / max(h - 1, 1)], axis=1)
        verts.append(np.stack([wx, wy, z_front], axis=1))
        uvs.append(uv)

    n_ring = contour_points
    front_counts = [1] + [n_ring] * rings
    front_offsets = np.cumsum([0] + front_counts[:-1])
    front_v = np.concatenate(verts).astype(np.float32)
    front_uv = np.concatenate(uvs).astype(np.float32)

    back_v = front_v.copy()
    back_v[:, 2] *= -back_depth_ratio
    back_uv = front_uv.copy()

    all_v = np.concatenate([front_v, back_v]).astype(np.float32)
    all_uv = np.concatenate([front_uv, back_uv]).astype(np.float32)
    back_base = len(front_v)

    faces = []
    apex = int(front_offsets[0])
    r1 = int(front_offsets[1])
    for i in range(n_ring):
        j = (i + 1) % n_ring
        faces.append([apex, r1 + i, r1 + j])
    for r in range(1, rings):
        a0, b0 = int(front_offsets[r]), int(front_offsets[r + 1])
        for i in range(n_ring):
            j = (i + 1) % n_ring
            faces.append([a0 + i, b0 + i, b0 + j])
            faces.append([a0 + i, b0 + j, a0 + j])

    b_apex = back_base + apex
    b_r1 = back_base + r1
    for i in range(n_ring):
        j = (i + 1) % n_ring
        faces.append([b_apex, b_r1 + j, b_r1 + i])
    for r in range(1, rings):
        a0 = back_base + int(front_offsets[r])
        b0 = back_base + int(front_offsets[r + 1])
        for i in range(n_ring):
            j = (i + 1) % n_ring
            faces.append([a0 + i, b0 + j, b0 + i])
            faces.append([a0 + i, a0 + j, b0 + j])

    faces = np.array(faces, dtype=np.int32)
    shell = orient_faces_outward(
        Mesh(all_v, faces, np.zeros_like(all_v), np.zeros_like(all_v)))
    faces = shell.faces
    normals = _vertex_normals(all_v, faces)

    # Centre on x/z and put the base at y = -height/2, regardless of where the
    # mask centroid happened to fall inside the crop.
    all_v[:, 0] -= 0.5 * (float(all_v[:, 0].min()) + float(all_v[:, 0].max()))
    all_v[:, 2] -= 0.5 * (float(all_v[:, 2].min()) + float(all_v[:, 2].max()))
    height = float(all_v[:, 1].max()) - float(all_v[:, 1].min())
    all_v[:, 1] -= float(all_v[:, 1].min()) + height * 0.5

    colors = np.ones((len(all_v), 3), dtype=np.float32)
    return Mesh(all_v, faces, normals, colors, all_uv, sprite.copy(), _OBJECT_MATERIAL)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (smooth shading)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lens, 1e-8)).astype(np.float32)


def fallback_object_mesh(longest_dim_m: float,
                         color=(0.62, 0.60, 0.58)) -> Mesh:
    """Plain box stand-in, used when no photo-crop of the object exists yet."""
    from src.simulation.render import primitives as P

    longest = max(float(longest_dim_m), 0.02)
    size = (longest, longest * 0.62, longest * 0.40)
    return P.box((0.0, 0.0, 0.0), size, color, material=_OBJECT_MATERIAL)


def mesh_extent(mesh: Mesh) -> np.ndarray:
    """Axis-aligned (sx, sy, sz) of a mesh, in its own frame."""
    v = mesh.vertices
    return (v.max(axis=0) - v.min(axis=0)).astype(np.float32)
