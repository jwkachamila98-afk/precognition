"""Procedural mesh primitives in LAB WORLD space (+X right, +Y up, +Z viewer).

Back-face culling needs CCW-when-seen-from-outside winding. Rather than trust
each builder to have got its loop order right - three of them originally had it
backwards, which showed up as a pedestal rendering its own underside and
z-fighting the bench - every CLOSED solid runs its faces through
``orient_faces_outward`` before returning. Open surfaces (quad, ring) cannot be
oriented that way and are drawn double-sided instead.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from src.simulation.render.raster import Material, Mesh, orient_faces_outward


def _as_color(color) -> np.ndarray:
    return np.asarray(color, dtype=np.float32).reshape(3)


def quad(p0, p1, p2, p3, color, uvs: Optional[Sequence] = None,
         texture: Optional[np.ndarray] = None,
         material: Optional[Material] = None) -> Mesh:
    """A flat quad from 4 CCW corners seen from its front face."""
    v = np.array([p0, p1, p2, p3], dtype=np.float32)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    normals = np.tile(n, (4, 1)).astype(np.float32)
    colors = np.tile(_as_color(color), (4, 1))
    uv = np.array(uvs if uvs is not None else [[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Mesh(v, faces, normals, colors, uv, texture,
                material or Material(cull_backfaces=False))


def box(center, size, color, material: Optional[Material] = None,
        texture: Optional[np.ndarray] = None, uv_scale: float = 1.0) -> Mesh:
    """Axis-aligned box. Flat-shaded: each face carries its own duplicated verts."""
    c = np.asarray(center, dtype=np.float32)
    hx, hy, hz = np.asarray(size, dtype=np.float32) / 2.0
    p = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32) + c

    face_loops = [
        ([4, 5, 6, 7], (0, 0, 1)),      # +Z
        ([1, 0, 3, 2], (0, 0, -1)),     # -Z
        ([5, 1, 2, 6], (1, 0, 0)),      # +X
        ([0, 4, 7, 3], (-1, 0, 0)),     # -X
        ([7, 6, 2, 3], (0, 1, 0)),      # +Y
        ([0, 1, 5, 4], (0, -1, 0)),     # -Y
    ]
    verts, norms, faces, uvs = [], [], [], []
    base_uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32) * uv_scale
    for loop, n in face_loops:
        off = len(verts)
        for i in loop:
            verts.append(p[i])
            norms.append(n)
        uvs.extend(base_uv)
        faces.append([off, off + 1, off + 2])
        faces.append([off, off + 2, off + 3])

    verts = np.array(verts, dtype=np.float32)
    return orient_faces_outward(Mesh(
        verts,
        np.array(faces, dtype=np.int32),
        np.array(norms, dtype=np.float32),
        np.tile(_as_color(color), (len(verts), 1)),
        np.array(uvs, dtype=np.float32),
        texture,
        material or Material(),
    ))


def cylinder(center, radius: float, height: float, color, segments: int = 16,
             axis: str = "y", caps: bool = True,
             material: Optional[Material] = None) -> Mesh:
    """Smooth-shaded cylinder with optional flat caps."""
    c = np.asarray(center, dtype=np.float32)
    hh = height / 2.0
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    ca, sa = np.cos(ang), np.sin(ang)

    if axis == "y":
        ring = np.stack([ca * radius, np.zeros_like(ca), sa * radius], axis=1)
        rn = np.stack([ca, np.zeros_like(ca), sa], axis=1)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif axis == "x":
        ring = np.stack([np.zeros_like(ca), ca * radius, sa * radius], axis=1)
        rn = np.stack([np.zeros_like(ca), ca, sa], axis=1)
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        ring = np.stack([ca * radius, sa * radius, np.zeros_like(ca)], axis=1)
        rn = np.stack([ca, sa, np.zeros_like(ca)], axis=1)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    bottom = c - up * hh + ring
    top = c + up * hh + ring
    verts = [bottom, top]
    norms = [rn, rn]
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        b_i, b_j = i, j
        t_i, t_j = segments + i, segments + j
        faces.append([b_i, b_j, t_j])
        faces.append([b_i, t_j, t_i])

    if caps:
        off = 2 * segments
        verts.append(top)
        norms.append(np.tile(up, (segments, 1)))
        verts.append(np.array([c + up * hh], dtype=np.float32))
        norms.append(up[None, :])
        top_center = off + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([top_center, off + j, off + i])

        off2 = top_center + 1
        verts.append(bottom)
        norms.append(np.tile(-up, (segments, 1)))
        verts.append(np.array([c - up * hh], dtype=np.float32))
        norms.append(-up[None, :])
        bot_center = off2 + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([bot_center, off2 + i, off2 + j])

    v = np.concatenate(verts).astype(np.float32)
    n = np.concatenate(norms).astype(np.float32)
    return orient_faces_outward(
        Mesh(v, np.array(faces, dtype=np.int32), n,
             np.tile(_as_color(color), (len(v), 1)),
             material=material or Material()))


def ring(center, inner_radius: float, outer_radius: float, color,
         segments: int = 32, material: Optional[Material] = None) -> Mesh:
    """Flat annulus in the XZ plane, facing +Y. Used for inlaid light strips -
    a filled disc reads as a plate, an annulus reads as a lit edge."""
    c = np.asarray(center, dtype=np.float32)
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    ca, sa = np.cos(ang), np.sin(ang)
    inner = c + np.stack([ca * inner_radius, np.zeros_like(ca), sa * inner_radius], axis=1)
    outer = c + np.stack([ca * outer_radius, np.zeros_like(ca), sa * outer_radius], axis=1)
    v = np.concatenate([inner, outer]).astype(np.float32)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, segments + i, segments + j])
        faces.append([i, segments + j, j])
    n = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(v), 1))
    return Mesh(v, np.array(faces, dtype=np.int32), n,
                np.tile(_as_color(color), (len(v), 1)),
                material=material or Material(cull_backfaces=False))


def uv_sphere(center, radius: float, color, segments: int = 10, rings: int = 6,
              material: Optional[Material] = None) -> Mesh:
    """Smooth-shaded UV sphere."""
    c = np.asarray(center, dtype=np.float32)
    verts, norms, faces = [], [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            n = np.array([
                math.sin(phi) * math.cos(theta),
                math.cos(phi),
                math.sin(phi) * math.sin(theta),
            ], dtype=np.float32)
            norms.append(n)
            verts.append(c + n * radius)

    for r in range(rings):
        for s in range(segments):
            s2 = (s + 1) % segments
            a = r * segments + s
            b = r * segments + s2
            d = (r + 1) * segments + s
            e = (r + 1) * segments + s2
            faces.append([a, e, b])
            faces.append([a, d, e])

    v = np.array(verts, dtype=np.float32)
    return orient_faces_outward(
        Mesh(v, np.array(faces, dtype=np.int32), np.array(norms, dtype=np.float32),
             np.tile(_as_color(color), (len(v), 1)),
             material=material or Material()))


def lathe(profile, color, segments: int = 20, close_bottom: bool = True,
          close_top: bool = True, material: Optional[Material] = None) -> Mesh:
    """Surface of revolution about +Y from a (radius, height) profile.

    Nearly everything a hand picks up off a bench is turned: bottles, cups,
    cans, glasses, bowls, fruit. Given the object's class we know its profile,
    which produces real geometry instead of guessing depth from a silhouette.

    `profile` runs bottom to top as [(radius, y), ...]; a zero radius at either
    end closes that end to a point, so a cap is expressed in the profile itself
    rather than as a separate primitive.
    """
    prof = np.asarray(profile, dtype=np.float32).reshape(-1, 2)
    if len(prof) < 2:
        raise ValueError("lathe: profile needs at least two points")

    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    ca, sa = np.cos(ang), np.sin(ang)

    rings, normals = [], []
    for i, (r, y) in enumerate(prof):
        rings.append(np.stack([ca * r, np.full(segments, y, np.float32), sa * r], axis=1))
        # Normal is perpendicular to the profile tangent, revolved outward, so a
        # shoulder or a taper shades as a slope rather than a hard step.
        j0, j1 = max(i - 1, 0), min(i + 1, len(prof) - 1)
        dr, dy = prof[j1][0] - prof[j0][0], prof[j1][1] - prof[j0][1]
        n_len = max(math.hypot(dr, dy), 1e-8)
        nr, ny = dy / n_len, -dr / n_len
        normals.append(np.stack([ca * nr, np.full(segments, ny, np.float32), sa * nr], axis=1))

    verts = list(rings)
    norms = list(normals)
    faces = []
    for i in range(len(prof) - 1):
        for s_i in range(segments):
            s_j = (s_i + 1) % segments
            a = i * segments + s_i
            b = i * segments + s_j
            c = (i + 1) * segments + s_i
            d = (i + 1) * segments + s_j
            faces.append([a, b, d])
            faces.append([a, d, c])

    off = len(prof) * segments
    for close, idx, up in ((close_bottom, 0, -1.0), (close_top, len(prof) - 1, 1.0)):
        if not close or prof[idx][0] <= 1e-6:
            continue
        centre = np.array([[0.0, prof[idx][1], 0.0]], dtype=np.float32)
        verts.append(centre)
        norms.append(np.array([[0.0, up, 0.0]], dtype=np.float32))
        ring0 = idx * segments
        for s_i in range(segments):
            s_j = (s_i + 1) % segments
            faces.append([off, ring0 + s_i, ring0 + s_j])
        off += 1

    v = np.concatenate(verts).astype(np.float32)
    n = np.concatenate(norms).astype(np.float32)
    return orient_faces_outward(
        Mesh(v, np.array(faces, dtype=np.int32), n,
             np.tile(_as_color(color), (len(v), 1)),
             material=material or Material()))


def _parallel_transport_frames(points: np.ndarray) -> tuple:
    """Rotation-minimising (U, V) frames along a polyline.

    Naively re-deriving a side vector per segment makes a swept tube twist
    visibly wherever the curve bends; parallel transport carries the previous
    frame forward instead, which is what keeps the finger tubes clean.
    """
    n = len(points)
    tangents = np.zeros((n, 3), dtype=np.float32)
    tangents[:-1] = points[1:] - points[:-1]
    tangents[-1] = tangents[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    lens = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lens, 1e-8)

    seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(tangents[0], seed))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(tangents[0], seed)
    u /= max(float(np.linalg.norm(u)), 1e-8)

    us = np.zeros((n, 3), dtype=np.float32)
    vs = np.zeros((n, 3), dtype=np.float32)
    us[0] = u
    vs[0] = np.cross(tangents[0], u)
    for i in range(1, n):
        prev_t, cur_t = tangents[i - 1], tangents[i]
        axis = np.cross(prev_t, cur_t)
        s = float(np.linalg.norm(axis))
        if s < 1e-7:
            u_i = us[i - 1]
        else:
            axis = axis / s
            angle = math.atan2(s, float(np.dot(prev_t, cur_t)))
            ca, sa = math.cos(angle), math.sin(angle)
            prev_u = us[i - 1]
            # Rodrigues rotation of the previous side vector onto the new tangent.
            u_i = prev_u * ca + np.cross(axis, prev_u) * sa + axis * float(np.dot(axis, prev_u)) * (1.0 - ca)
        u_i = u_i - tangents[i] * float(np.dot(u_i, tangents[i]))
        u_i /= max(float(np.linalg.norm(u_i)), 1e-8)
        us[i] = u_i
        vs[i] = np.cross(tangents[i], u_i)
    return us, vs, tangents


def tube(points, radii, color, segments: int = 6, cap_start: bool = True,
         cap_end: bool = True, material: Optional[Material] = None) -> Mesh:
    """Swept tube through a 3-D polyline with a per-point radius.

    Returns an empty mesh for degenerate input (fewer than 2 distinct points),
    which the rasterizer skips harmlessly.
    """
    pts = np.asarray(points, dtype=np.float32)
    rad = np.asarray(radii, dtype=np.float32).reshape(-1)
    if len(pts) < 2:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                    np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                    material=material or Material())

    us, vs, tangents = _parallel_transport_frames(pts)
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]

    ring_normals = ca[None, :, :] * us[:, None, :] + sa[None, :, :] * vs[:, None, :]  # (n, seg, 3)
    ring_verts = pts[:, None, :] + ring_normals * rad[:, None, None]

    n_pts = len(pts)
    verts = ring_verts.reshape(-1, 3)
    norms = ring_normals.reshape(-1, 3)
    faces = []
    for i in range(n_pts - 1):
        for s in range(segments):
            s2 = (s + 1) % segments
            a = i * segments + s
            b = i * segments + s2
            c = (i + 1) * segments + s
            d = (i + 1) * segments + s2
            faces.append([a, b, d])
            faces.append([a, d, c])

    verts = [verts]
    norms = [norms]
    off = n_pts * segments
    if cap_end:
        verts.append(pts[-1][None, :])
        norms.append(tangents[-1][None, :])
        cidx = off
        for s in range(segments):
            s2 = (s + 1) % segments
            faces.append([cidx, (n_pts - 1) * segments + s, (n_pts - 1) * segments + s2])
        off += 1
    if cap_start:
        verts.append(pts[0][None, :])
        norms.append(-tangents[0][None, :])
        cidx = off
        for s in range(segments):
            s2 = (s + 1) % segments
            faces.append([cidx, s2, s])

    v = np.concatenate(verts).astype(np.float32)
    n = np.concatenate(norms).astype(np.float32)
    return orient_faces_outward(
        Mesh(v, np.array(faces, dtype=np.int32), n,
             np.tile(_as_color(color), (len(v), 1)),
             material=material or Material()))


def prism(polygon_xy, thickness: float, color, plane_origin=None, plane_u=None,
          plane_v=None, plane_n=None, material: Optional[Material] = None) -> Mesh:
    """Extrude a 2-D polygon (CCW in its own plane) into a slab of `thickness`.

    Used for the palm: a hand's palm is a flat-ish shell, not a bundle of
    cylinders, so extruding its outline gives far better silhouette than
    capsules would.
    """
    poly = np.asarray(polygon_xy, dtype=np.float32)
    k = len(poly)
    if k < 3:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                    np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                    material=material or Material())

    origin = np.zeros(3, np.float32) if plane_origin is None else np.asarray(plane_origin, np.float32)
    u = np.array([1, 0, 0], np.float32) if plane_u is None else np.asarray(plane_u, np.float32)
    v = np.array([0, 1, 0], np.float32) if plane_v is None else np.asarray(plane_v, np.float32)
    n = np.cross(u, v) if plane_n is None else np.asarray(plane_n, np.float32)
    n = n / max(float(np.linalg.norm(n)), 1e-9)

    mid = origin + poly[:, 0:1] * u + poly[:, 1:2] * v
    half = n * (thickness / 2.0)
    front = mid + half
    back = mid - half

    verts = [front, back]
    norms = [np.tile(n, (k, 1)), np.tile(-n, (k, 1))]
    faces = []
    centroid_f = front.mean(axis=0)
    centroid_b = back.mean(axis=0)
    verts.append(np.array([centroid_f, centroid_b], dtype=np.float32))
    norms.append(np.array([n, -n], dtype=np.float32))
    cf, cb = 2 * k, 2 * k + 1
    for i in range(k):
        j = (i + 1) % k
        faces.append([cf, i, j])                    # front cap
        faces.append([cb, k + j, k + i])            # back cap

    # Side wall: its own duplicated verts so the rim shades as a hard edge.
    off = 2 * k + 2
    side_v, side_n = [], []
    for i in range(k):
        j = (i + 1) % k
        edge = mid[j] - mid[i]
        sn = np.cross(edge, n)
        sn = sn / max(float(np.linalg.norm(sn)), 1e-9)
        base = off + 4 * i
        side_v.extend([front[i], front[j], back[j], back[i]])
        side_n.extend([sn, sn, sn, sn])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    verts.append(np.array(side_v, dtype=np.float32))
    norms.append(np.array(side_n, dtype=np.float32))

    vv = np.concatenate(verts).astype(np.float32)
    nn = np.concatenate(norms).astype(np.float32)
    return orient_faces_outward(
        Mesh(vv, np.array(faces, dtype=np.int32), nn,
             np.tile(_as_color(color), (len(vv), 1)),
             material=material or Material(cull_backfaces=False)))
