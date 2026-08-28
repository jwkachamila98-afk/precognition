"""A z-buffered, perspective-correct triangle rasterizer in pure numpy.

Why hand-rolled: this project targets a CPU-only Intel Mac at 30+ FPS with a
dependency set that must stay installable (no CUDA, no native GL context). Every
off-the-shelf option (pyrender, moderngl, Open3D) drags in an offscreen GL
context that is historically fragile on exactly this hardware. numpy + cv2 are
already hard dependencies, so this adds nothing to install and cannot break the
existing demo.

Design: **deferred shading**. The per-triangle loop writes only geometry and
material attributes into a G-buffer; all the expensive lighting math (normalise,
dot, pow) then runs ONCE as a single vectorised full-screen pass in
``shading.py``. That keeps the hot loop to ~20 numpy calls per triangle and
makes the cost of extra lights essentially free.

Two further things carry the frame budget:
  * static lab geometry is rasterised once and its G-buffer cached
    (``Rasterizer.snapshot`` / ``restore``), so only the hand and the object are
    re-rasterised per frame;
  * attributes are interpolated only at the pixels that actually pass the depth
    test, as flat 1-D arrays, never over the whole triangle bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.simulation.render.camera import Camera

# G-buffer channel layout. Colours are BGR throughout (OpenCV's order) so the
# resolved image needs no conversion before it reaches cv2.
GB_ALBEDO = slice(0, 3)
GB_NORMAL = slice(3, 6)
GB_WORLD = slice(6, 9)
GB_SPEC = 9
GB_SHIN = 10
GB_EMIS = 11
GB_COVER = 12
GB_CHANNELS = 13

# Vertex attributes interpolated across a triangle. Deliberately laid out in the
# same order as the first nine G-buffer channels (albedo, normal, world) so the
# untextured path copies one contiguous block instead of three slices.
_ATTR_COLOR = slice(0, 3)
_ATTR_NORMAL = slice(3, 6)
_ATTR_WORLD = slice(6, 9)
_ATTR_UV = slice(9, 11)
_ATTR_GEOM = slice(0, 9)
_ATTR_CHANNELS = 11

# Sign of the screen-space signed area for a front-facing (CCW-in-world)
# triangle. Screen Y points down while world Y points up, so the handedness
# flips and CCW geometry rasterises with a negative area. Verified by
# tests/test_lab_renderer.py::test_backface_culling_orientation.
FRONT_FACE_SIGN = -1.0


@dataclass
class Material:
    """Surface response used by the deferred shading pass."""

    specular: float = 0.15
    shininess: float = 24.0
    emissive: float = 0.0
    cull_backfaces: bool = True
    bilinear: bool = False


@dataclass
class Mesh:
    """A triangle mesh in LAB WORLD coordinates (see camera.py for the frame)."""

    vertices: np.ndarray                      # (V, 3) float32
    faces: np.ndarray                         # (F, 3) int32
    normals: np.ndarray                       # (V, 3) float32, unit length
    colors: np.ndarray                        # (V, 3) float32 BGR in [0, 1]
    uvs: Optional[np.ndarray] = None          # (V, 2) float32 in [0, 1]
    texture: Optional[np.ndarray] = None      # (th, tw, 3) uint8 BGR
    material: Material = field(default_factory=Material)

    @property
    def num_faces(self) -> int:
        return int(len(self.faces))

    def transformed(self, rotation: Optional[np.ndarray] = None,
                    translation: Optional[np.ndarray] = None,
                    scale: float = 1.0) -> "Mesh":
        """A copy with a rigid (optionally uniformly scaled) transform applied.

        Uniform scale only, so normals rotate with the same matrix as positions.
        """
        v = self.vertices * float(scale)
        n = self.normals
        if rotation is not None:
            R = np.asarray(rotation, dtype=np.float32)
            v = v @ R.T
            n = n @ R.T
        if translation is not None:
            v = v + np.asarray(translation, dtype=np.float32)
        return Mesh(v.astype(np.float32), self.faces, n.astype(np.float32), self.colors,
                    self.uvs, self.texture, self.material)


def concat_meshes(meshes) -> Mesh:
    """Merge untextured meshes sharing one material into a single draw batch.

    Textured meshes are deliberately rejected: a merged mesh carries exactly one
    texture, so silently keeping the first would mis-texture the rest.
    """
    meshes = [m for m in meshes if m is not None and m.num_faces > 0]
    if not meshes:
        raise ValueError("concat_meshes: nothing to merge")
    if any(m.texture is not None for m in meshes):
        raise ValueError("concat_meshes: textured meshes must be drawn separately")

    verts, faces, norms, cols = [], [], [], []
    offset = 0
    for m in meshes:
        verts.append(m.vertices)
        norms.append(m.normals)
        cols.append(m.colors)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return Mesh(
        np.concatenate(verts).astype(np.float32),
        np.concatenate(faces).astype(np.int32),
        np.concatenate(norms).astype(np.float32),
        np.concatenate(cols).astype(np.float32),
        material=meshes[0].material,
    )


def orient_faces_outward(mesh: Mesh) -> Mesh:
    """Rewind every face of a closed mesh so its front side faces outward.

    Back-face culling halves the rasterised triangle count, but only if winding
    is consistent - and for procedurally swept geometry (a palm slab whose
    outline order depends on which hand it is, a tube whose frame depends on the
    curve) consistency is not something the builder can guarantee. Testing each
    face normal against the direction from the mesh centroid settles it for the
    star-shaped-about-the-centroid solids used here: tubes, slabs, and the
    inflated object shell.
    """
    v, f = mesh.vertices, mesh.faces
    if len(f) == 0:
        return mesh
    centroid = v.mean(axis=0)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_normal = np.cross(v1 - v0, v2 - v0)
    outward = (v0 + v1 + v2) / 3.0 - centroid
    flip = np.sum(face_normal * outward, axis=1) < 0.0
    if not flip.any():
        return mesh
    faces = f.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(v, faces, mesh.normals, mesh.colors, mesh.uvs, mesh.texture, mesh.material)


def _clip_triangle_near(tri_view: np.ndarray, tri_attr: np.ndarray, near: float):
    """Sutherland-Hodgman clip of one triangle against the near plane.

    Operates in view space where the visible half-space is ``-z >= near``.
    Returns a list of (3, 3) / (3, A) fans - 0, 1, or 2 triangles.
    """
    poly_v, poly_a = [], []
    for i in range(3):
        j = (i + 1) % 3
        vi, vj = tri_view[i], tri_view[j]
        di, dj = -vi[2] - near, -vj[2] - near      # signed distance, positive = inside
        if di >= 0:
            poly_v.append(vi)
            poly_a.append(tri_attr[i])
        if (di >= 0) != (dj >= 0):
            t = di / (di - dj)
            poly_v.append(vi + t * (vj - vi))
            poly_a.append(tri_attr[i] + t * (tri_attr[j] - tri_attr[i]))

    if len(poly_v) < 3:
        return []
    out = []
    for k in range(1, len(poly_v) - 1):
        out.append((
            np.stack([poly_v[0], poly_v[k], poly_v[k + 1]]).astype(np.float32),
            np.stack([poly_a[0], poly_a[k], poly_a[k + 1]]).astype(np.float32),
        ))
    return out


def _sample_texture(texture: np.ndarray, uv: np.ndarray, bilinear: bool) -> np.ndarray:
    """Sample a BGR uint8 texture at (N, 2) UVs -> (N, 3) float in [0, 1].

    UV origin is top-left; v increases downward, matching image row order.
    """
    th, tw = texture.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0) * (tw - 1)
    v = np.clip(uv[:, 1], 0.0, 1.0) * (th - 1)
    if not bilinear:
        return texture[np.rint(v).astype(np.int32), np.rint(u).astype(np.int32)] * (1.0 / 255.0)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, tw - 1)
    y1 = np.minimum(y0 + 1, th - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    top = texture[y0, x0] * (1.0 - fx) + texture[y0, x1] * fx
    bot = texture[y1, x0] * (1.0 - fx) + texture[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy) * (1.0 / 255.0)


class Rasterizer:
    """G-buffer rasterizer. One instance owns one render target."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.gbuffer = np.zeros((self.height, self.width, GB_CHANNELS), dtype=np.float32)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        self.triangles_drawn = 0
        # Pixel-centre ramps, sliced per triangle as views - allocating these in
        # the hot loop cost more than the arithmetic they feed.
        self._xs = np.arange(self.width, dtype=np.float32) + 0.5
        self._ys = np.arange(self.height, dtype=np.float32) + 0.5

    def clear(self) -> None:
        self.gbuffer.fill(0.0)
        self.depth.fill(np.inf)
        self.triangles_drawn = 0

    def snapshot(self) -> tuple:
        """Copy of the render target, for caching a static scene."""
        return self.gbuffer.copy(), self.depth.copy()

    def restore(self, snap: tuple) -> None:
        """Reset the render target to a previously taken snapshot."""
        gb, d = snap
        np.copyto(self.gbuffer, gb)
        np.copyto(self.depth, d)
        self.triangles_drawn = 0

    def seed_depth(self, depth: np.ndarray) -> None:
        """Seed the depth buffer from another pass without inheriting its colour.

        Used for the translucent hologram layer: the ghost hand must be occluded
        by opaque lab geometry, but must not inherit the lab's shaded pixels.
        """
        np.copyto(self.depth, depth)

    def draw(self, mesh: Mesh, camera: Camera) -> None:
        """Rasterize one mesh into the G-buffer."""
        if mesh.num_faces == 0:
            return

        mat = mesh.material
        view = camera.to_view(mesh.vertices)

        uvs = mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.vertices), 2), dtype=np.float32)
        attrs = np.concatenate(
            [mesh.colors, mesh.normals, mesh.vertices, uvs], axis=1
        ).astype(np.float32)

        tri_view = view[mesh.faces]                    # (F, 3, 3)
        tri_attr = attrs[mesh.faces]                   # (F, 3, A)

        inside = (-tri_view[:, :, 2]) >= camera.near
        n_inside = inside.sum(axis=1)

        batches = []
        full = n_inside == 3
        if full.any():
            batches.append((tri_view[full], tri_attr[full]))

        partial_idx = np.flatnonzero((n_inside == 1) | (n_inside == 2))
        if len(partial_idx) > 0:
            cv, ca = [], []
            for i in partial_idx:
                for t_v, t_a in _clip_triangle_near(tri_view[i], tri_attr[i], camera.near):
                    cv.append(t_v)
                    ca.append(t_a)
            if cv:
                batches.append((np.stack(cv), np.stack(ca)))

        for b_view, b_attr in batches:
            self._raster_batch(b_view, b_attr, camera, mat, mesh.texture)

    def _raster_batch(self, tri_view, tri_attr, camera: Camera, mat: Material,
                      texture: Optional[np.ndarray]) -> None:
        n_tri = len(tri_view)
        flat_screen, flat_depth = camera.project_view(tri_view.reshape(-1, 3))
        scr = flat_screen.reshape(n_tri, 3, 2)
        dep = flat_depth.reshape(n_tri, 3)

        x0, y0 = scr[:, 0, 0], scr[:, 0, 1]
        x1, y1 = scr[:, 1, 0], scr[:, 1, 1]
        x2, y2 = scr[:, 2, 0], scr[:, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        keep = np.abs(area) > 1e-7
        if mat.cull_backfaces:
            keep &= (area * FRONT_FACE_SIGN) > 0

        # Reject triangles whose screen bounding box misses the viewport entirely.
        bx_min = np.minimum(np.minimum(x0, x1), x2)
        bx_max = np.maximum(np.maximum(x0, x1), x2)
        by_min = np.minimum(np.minimum(y0, y1), y2)
        by_max = np.maximum(np.maximum(y0, y1), y2)
        keep &= (bx_max >= 0) & (bx_min <= self.width - 1)
        keep &= (by_max >= 0) & (by_min <= self.height - 1)

        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return

        inv_area = (1.0 / area[idx]).astype(np.float32)
        inv_w = (1.0 / dep[idx]).astype(np.float32)          # (n, 3)
        ix0 = np.maximum(np.floor(bx_min[idx]).astype(np.int32), 0)
        ix1 = np.minimum(np.ceil(bx_max[idx]).astype(np.int32), self.width - 1)
        iy0 = np.maximum(np.floor(by_min[idx]).astype(np.int32), 0)
        iy1 = np.minimum(np.ceil(by_max[idx]).astype(np.int32), self.height - 1)

        s = scr[idx]
        a = tri_attr[idx]
        gb, zb = self.gbuffer, self.depth
        spec, shin, emis = mat.specular, mat.shininess, mat.emissive
        textured = texture is not None

        # Per-triangle coefficient maths runs on PYTHON floats, not numpy
        # scalars: a numpy float32 scalar op carries ~1 us of dispatch, and
        # there are a dozen of them per triangle.
        s_list = s.tolist()
        w_list = inv_w.tolist()
        ia_list = inv_area.tolist()
        bounds = np.stack([ix0, ix1, iy0, iy1], axis=1).tolist()
        const_tail = np.array([spec, shin, emis, 1.0], dtype=np.float32)
        xs_all, ys_all = self._xs, self._ys

        for k in range(len(idx)):
            px0, px1, py0, py1 = bounds[k]
            xs = xs_all[px0:px1 + 1]
            ys = ys_all[py0:py1 + 1]

            (ax, ay), (bx, by), (cx, cy) = s_list[k]
            ia = ia_list[k]
            w0, w1, w2 = w_list[k]

            # Barycentrics and 1/depth are all AFFINE in (x, y), so each is
            # evaluated as `f(x) + g(y)` from two 1-D arrays and one broadcast
            # add. Doing the algebra here instead of multiplying full 2-D grids
            # is the single biggest saving in the rasterizer.
            a0x, a0y, a0c = (by - cy) * ia, (cx - bx) * ia, (bx * cy - cx * by) * ia
            a1x, a1y, a1c = (cy - ay) * ia, (ax - cx) * ia, (cx * ay - ax * cy) * ia

            l0 = (a0x * xs)[None, :] + (a0y * ys + a0c)[:, None]
            l1 = (a1x * xs)[None, :] + (a1y * ys + a1c)[:, None]

            # denom = l0*w0 + l1*w1 + (1-l0-l1)*w2, affine in (x, y) as well.
            d0, d1 = w0 - w2, w1 - w2
            dx = a0x * d0 + a1x * d1
            dy = a0y * d0 + a1y * d1
            dc = a0c * d0 + a1c * d1 + w2
            denom = (dx * xs)[None, :] + (dy * ys + dc)[:, None]
            np.maximum(denom, 1e-9, out=denom)
            z = np.reciprocal(denom)

            zroi = zb[py0:py1 + 1, px0:px1 + 1]
            covered = (l0 >= 0.0) & (l1 >= 0.0) & ((l0 + l1) <= 1.0) & (z < zroi)
            if not covered.any():
                continue

            # Everything below runs on 1-D arrays of covered pixels only.
            zc = z[covered]
            b0 = l0[covered] * (w0 * zc)
            b1 = l1[covered] * (w1 * zc)
            weights = np.empty((len(zc), 3), dtype=np.float32)
            weights[:, 0] = b0
            weights[:, 1] = b1
            weights[:, 2] = 1.0 - b0 - b1
            interp = weights @ a[k]                      # (n, A), BLAS-backed

            out = np.empty((len(zc), GB_CHANNELS), dtype=np.float32)
            out[:, _ATTR_GEOM] = interp[:, _ATTR_GEOM]
            if textured:
                out[:, GB_ALBEDO] *= _sample_texture(texture, interp[:, _ATTR_UV],
                                                     mat.bilinear)
            out[:, GB_SPEC:] = const_tail

            gb[py0:py1 + 1, px0:px1 + 1][covered] = out
            zroi[covered] = zc

        self.triangles_drawn += len(idx)
