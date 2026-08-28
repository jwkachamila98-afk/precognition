"""The simulated lab: renders the foreseen trajectory as a reenactment in 3-D.

Replaces the flat 2-D ghost-hand overlay for the Autonomous Demo. Instead of
compositing capsule bones onto the webcam image, the planned trajectory is
staged inside an actual rendered robotics lab - workbench, backdrop, light rig -
with the real target object reconstructed from its own photo-crop and a
holographic hand mesh executing the plan.

FRAMES. Two coordinate systems meet here and must not be mixed:
  * PERCEPTION frame (everything upstream: BoundingBox3D, ForeseenWaypoint) -
    +X right, +Y DOWN, +Z away from the camera, metres.
  * LAB WORLD frame (everything in src/simulation/render) - +X right, +Y UP,
    +Z toward the viewer, metres, floor at y = 0.
``LabTransform`` is the single conversion between them.

BUDGET. The static lab (~350 triangles, large-area) is rasterized once per demo
and its G-buffer cached; only the object (~280 tris) and the hand (~390 tris)
are re-rasterized per frame, then shaded in one vectorised full-screen pass.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.perception.scene_parser import BoundingBox3D
from src.simulation.render import hand_mesh as HM
from src.simulation.render import lab_scene as LS
from src.simulation.render import object_mesh as OM
from src.simulation.render import shading as SH
from src.simulation.render.camera import Camera
from src.simulation.render.raster import Material, Mesh, Rasterizer
from src.simulation.trajectory_generator import ForeseenTrajectory

logger = logging.getLogger(__name__)

# Perception (+Y down, +Z away) -> lab world (+Y up, +Z toward viewer).
_AXIS_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

# Anthropometric reference: adult wrist -> middle-MCP distance. Trajectory
# keypoints are rescaled to this so the hand has believable proportions
# regardless of the tracker's own metric calibration.
_REFERENCE_PALM_M = 0.092

# Fixed staging angles for the lab camera. Declared here rather than inside
# _fit_camera because the object's orientation is derived from the same view
# direction and must be settled before the camera (which needs the object's
# final height) can be framed.
_CAM_AZ_DEG = 17.0
_CAM_EL_DEG = 21.0

# The lab camera is LOCKED: one pose, identical for every demo, regardless of
# object or plan. Framing the shot adaptively made each reenactment look like a
# different scene - the viewer had to re-read the geometry every time before
# they could see what the hand was doing. A fixed pose means the only thing that
# changes between runs is the thing being demonstrated.
#
# It is chosen to hold the whole staged action: the manipuland (bounded to at
# most ~20 cm by the hand-size guard), the wrist at full lift, and enough bench
# and backdrop to place them. A very small object underfills it rather than
# pulling the camera in, which is the deliberate trade for consistency.
_CAM_DISTANCE_M = 0.87
_CAM_TARGET_HEIGHT_M = 0.20      # above the staging pedestal
_CAM_FOV_Y_DEG = 42.0


def _view_direction() -> np.ndarray:
    """Unit vector from the staging area toward the lab camera."""
    az, el = math.radians(_CAM_AZ_DEG), math.radians(_CAM_EL_DEG)
    return np.array([
        math.sin(az) * math.cos(el),
        math.sin(el),
        math.cos(az) * math.cos(el),
    ], dtype=np.float32)

# Thumb, index, middle, ring, pinky tips in the 21-joint layout.
_FINGERTIPS = [4, 8, 12, 16, 20]

_HOLOGRAM_COLOR = np.array([1.00, 0.70, 0.10], dtype=np.float32)   # BGR: electric cyan
_HOLOGRAM_MATERIAL = Material(specular=0.32, shininess=42.0, emissive=0.85)


@dataclass
class LabTransform:
    """Maps perception-frame points into the lab, anchored on the object."""

    origin_cam: np.ndarray      # object centre in the perception frame
    anchor_lab: np.ndarray      # where that point lands in the lab
    scale: float

    def __call__(self, points_cam: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float32))
        out = self.anchor_lab + ((p - self.origin_cam) @ _AXIS_FLIP.T) * self.scale
        return out.astype(np.float32)


# A manipuland is only as big as the hand grasping it can plausibly handle.
# Expressed as multiples of palm length (wrist -> middle MCP, ~9.2 cm).
_MIN_OBJECT_PALMS = 0.45
_MAX_OBJECT_PALMS = 2.2


def _object_longest_dimension(bbox: Optional[BoundingBox3D],
                              hand_palm_m: float) -> float:
    """The manipuland's longest dimension in metres, in a scale the plan supports.

    Only the SCALAR is taken from the detector - its aspect comes from the
    silhouette instead (see object_mesh.build_object_mesh). Even the scalar is
    weak: it is back-projected from depth that is synthetic locally and only
    relative (MiDaS) on the GPU pod, so it is not metric on either path. In
    production it reported a wine glass at over 34 cm, which staged a beach-ball
    on the bench and forced the camera back until the reenactment was unreadable.

    So it is bounded by the one metric reference actually present in the scene:
    the hand doing the grasping. An object the hand could not close around is not
    the object this plan is grasping, whatever the detector says.
    """
    raw = None
    if bbox is not None:
        size = np.asarray(bbox.size, dtype=np.float32).reshape(3)
        if np.all(np.isfinite(size)) and float(size.max()) > 1e-3:
            raw = float(size.max())
    if raw is None:
        return float(np.clip(1.4 * hand_palm_m, 0.05, 0.20))

    lo = _MIN_OBJECT_PALMS * hand_palm_m
    hi = _MAX_OBJECT_PALMS * hand_palm_m
    clamped = float(np.clip(raw, lo, hi))
    if abs(clamped - raw) > 1e-3:
        logger.info(
            f"LabSimulator: detector reported the target at {raw*100:.0f} cm, which the "
            f"grasping hand ({hand_palm_m*100:.1f} cm palm) could not handle; staging it "
            f"at {clamped*100:.0f} cm instead."
        )
    return clamped


class LabSimulator:
    """Renders the Autonomous Demo reenactment inside the simulated lab."""

    def __init__(self, width: int = 384, height: int = 288,
                 max_fps: float = 20.0) -> None:
        self.width = int(width)
        self.height = int(height)
        # The reenactment is a 2 s plan stretched over a ~6 s phase, so it has
        # far less than 30 fps of new information in it. Capping the render rate
        # keeps the client's own loop from paying for frames nobody can see.
        self._min_interval = 1.0 / max(float(max_fps), 1.0)
        self._last_render_t = 0.0
        self._last_image: Optional[np.ndarray] = None
        self._last_key = None
        self.scene = LS.build_lab()
        self._rast = Rasterizer(self.width, self.height)
        self._ghost = Rasterizer(self.width, self.height)
        self._static_snapshot = None
        self._static_camera_key = None
        self._static_lit: Optional[np.ndarray] = None      # tone-mapped BGR uint8
        self._static_depth: Optional[np.ndarray] = None

        self.camera: Optional[Camera] = None
        self.transform: Optional[LabTransform] = None
        self.trajectory: Optional[ForeseenTrajectory] = None
        self.object_mesh: Optional[Mesh] = None
        self.object_size = np.array([0.13, 0.08, 0.05], dtype=np.float32)
        self.object_is_reconstructed = False
        self.target_label = "object"
        self._hand_paths_lab: Optional[np.ndarray] = None      # (T, 21, 3)
        self._object_path_lab: Optional[np.ndarray] = None     # (T, 3)
        self._ready = False
        self.last_render_ms = 0.0

        self._lights, self._env = self._build_lighting()
        self._vignette = self._build_vignette()

    def _build_vignette(self) -> np.ndarray:
        """Radial falloff, baked into the static image and applied to every
        re-shaded pixel, so the eye lands on the staging area."""
        yy, xx = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        nx = (xx / (self.width - 1) - 0.5) * 2.0
        ny = (yy / (self.height - 1) - 0.5) * 2.0
        r2 = nx * nx + ny * ny
        return np.clip(1.0 - 0.30 * r2 * r2, 0.35, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _build_lighting():
        lights = [
            # Key: the overhead softbox bar, modelled as a point light so the
            # bench falls off toward the edges instead of lighting flatly.
            SH.Light(direction=(-0.22, -1.0, -0.28), color=(0.96, 0.98, 1.00),
                     intensity=1.15, point=(-0.30, 2.24, 0.10), radius=2.4),
            # Fill from the camera side: soft, warm, just enough to open shadows.
            SH.Light(direction=(0.55, -0.18, -1.0), color=(0.88, 0.80, 0.70),
                     intensity=0.26),
            # Rim from behind the backdrop, separating the actors from the wall.
            SH.Light(direction=(0.30, -0.28, 1.0), color=(1.00, 0.74, 0.34),
                     intensity=0.42),
        ]
        env = SH.Environment(
            sky_color=(0.150, 0.144, 0.138),
            ground_color=(0.048, 0.046, 0.050),
            fog_color=(0.040, 0.036, 0.034),
            fog_start=2.6, fog_end=8.5, fog_density=0.55,
            exposure=1.12,
        )
        return lights, env

    def prepare(self, trajectory: Optional[ForeseenTrajectory],
                target_bbox: Optional[BoundingBox3D],
                sprite: Optional[np.ndarray]) -> bool:
        """Stage a trajectory in the lab. Call once when the demo starts.

        Returns False if there is nothing renderable, in which case the caller
        should fall back to the 2-D overlay.
        """
        if trajectory is None or not trajectory.waypoints:
            self._ready = False
            return False

        t0 = time.perf_counter()
        waypoints = trajectory.waypoints
        self.trajectory = trajectory
        self.target_label = (trajectory.target_label or
                             (target_bbox.label if target_bbox else "object"))

        # Hand scale first: it is the metric reference the object is sized against.
        raw_kpts = np.stack([wp.hand_keypoints_3d for wp in waypoints]).astype(np.float32)
        measured_palm = float(np.median(
            np.linalg.norm(raw_kpts[:, HM._MCP["middle"]] - raw_kpts[:, HM.WRIST], axis=1)))
        scale = float(np.clip(_REFERENCE_PALM_M / max(measured_palm, 1e-4), 0.15, 8.0))
        longest = _object_longest_dimension(target_bbox, _REFERENCE_PALM_M)

        origin_cam = (np.asarray(target_bbox.center, dtype=np.float32) if target_bbox is not None
                      else np.asarray(waypoints[0].object_pose[:3], dtype=np.float32))
        mesh = None
        if sprite is not None and sprite.size > 0:
            try:
                mesh = OM.build_object_mesh(sprite, longest,
                                            contour_points=24, rings=2)
            except Exception as exc:                      # reconstruction is best-effort
                logger.warning(f"LabSimulator: object reconstruction failed ({exc}); using primitive.")
        self.object_is_reconstructed = mesh is not None
        if mesh is None:
            mesh = OM.fallback_object_mesh(longest)
        # Orient BEFORE measuring: turning the object changes its height, and the
        # anchor that rests it on the pedestal is derived from that height. Only
        # the RECONSTRUCTED mesh is turned - see _orient_to_camera.
        if self.object_is_reconstructed:
            mesh = self._orient_to_camera(mesh)
        else:
            mesh = self._centre_mesh(mesh)
        self.object_mesh = mesh
        self.object_size = OM.mesh_extent(mesh)

        anchor_lab = LS.OBJECT_ANCHOR + np.array(
            [0.0, float(self.object_size[1]) * 0.5, 0.0], dtype=np.float32)

        self.transform = LabTransform(origin_cam=origin_cam, anchor_lab=anchor_lab,
                                      scale=scale)
        self._hand_paths_lab = self.transform(raw_kpts.reshape(-1, 3)).reshape(raw_kpts.shape)
        self._object_path_lab = self.transform(
            np.stack([wp.object_pose[:3] for wp in waypoints]).astype(np.float32))

        # Keep the staged object from sinking through or floating above the bench:
        # the plan's own lift is preserved, its resting height is corrected.
        rest_y = float(anchor_lab[1])
        self._object_path_lab[:, 1] += rest_y - float(self._object_path_lab[0, 1])

        self._seat_grasp_on_the_staged_object()
        self.camera = self._fit_camera()
        self._ensure_static_background()
        self._ready = True

        logger.info(
            f"LabSimulator: staged '{self.target_label}' "
            f"({self.object_size[0]*100:.0f}x{self.object_size[1]*100:.0f}x"
            f"{self.object_size[2]*100:.0f} cm, hand scale x{scale:.2f}) across "
            f"{len(waypoints)} steps - lab {self.scene.triangle_count} tris static, "
            f"{self.object_mesh.num_faces} object tris, prepared in "
            f"{(time.perf_counter() - t0)*1000:.0f} ms."
        )
        return True

    def _seat_grasp_on_the_staged_object(self) -> None:
        """Re-seat the hand so its grasp lands on the object as STAGED.

        The planner sizes its approach from the detector's raw extent, but the
        lab stages the object at a corrected scale (see
        _object_longest_dimension). When those disagree - which is most of the
        time, since the detector's extent comes from non-metric depth - the hand
        closes in the air above a smaller object.

        Shifting the whole hand path by a constant offset preserves the plan
        exactly: the hand's motion relative to the object, and the object's own
        lift, are both unchanged. Only where the grasp sits is corrected.
        """
        grasp = self._contact_step()
        tips = self._hand_paths_lab[grasp][_FINGERTIPS]
        pinch = 0.5 * (tips[0] + tips[[1, 2]].mean(axis=0))
        # Grip the upper third of the object, where a hand actually takes a cup.
        target = self._object_path_lab[grasp] + np.array(
            [0.0, float(self.object_size[1]) * 0.18, 0.0], dtype=np.float32)
        self._hand_paths_lab += (target - pinch)[None, None, :]

    def invalidate(self) -> None:
        """Drop the staged trajectory so the next demo re-stages from scratch."""
        self._ready = False
        self.trajectory = None
        self._last_image = None
        self._last_key = None

    @property
    def is_ready(self) -> bool:
        return self._ready and self.camera is not None

    def _fit_camera(self) -> Camera:
        """The locked lab camera - the same pose for every reenactment.

        Deliberately ignores the trajectory. An earlier version fitted the shot
        to each plan's extent, which kept everything perfectly in frame but made
        no two demos look alike: the camera crept in and out between runs and a
        large manipuland pushed it back far enough to shrink the hand past
        legibility. Consistency is worth more here than optimal fill.

        The consequence to be aware of: a plan whose approach begins far from the
        bench (the user's real hand across the room, say) now starts off-screen
        and flies in. That reads fine - it is an entrance, not a glitch - but it
        does mean the viewport no longer guarantees the whole path is visible.
        """
        target = LS.OBJECT_ANCHOR + np.array([0.0, _CAM_TARGET_HEIGHT_M, 0.0],
                                             dtype=np.float32)
        position = target + _view_direction() * _CAM_DISTANCE_M
        return Camera(position=position, target=target, up=(0.0, 1.0, 0.0),
                      fov_y_deg=_CAM_FOV_Y_DEG, width=self.width, height=self.height,
                      near=0.04, far=40.0)

    def _contact_step(self) -> int:
        """The trajectory step where the grasp actually closes - the hero beat."""
        contacts = np.array([float(np.mean(wp.contact_state))
                             for wp in self.trajectory.waypoints], dtype=np.float32)
        closed = np.flatnonzero(contacts >= 0.98)
        return int(closed[0]) if len(closed) else int(np.argmax(contacts))

    def _orient_to_camera(self, mesh: Mesh) -> Mesh:
        """Turn the reconstructed object to face the lab camera, then seat it.

        The reconstruction is view-aligned relief: real silhouette and real
        surface colour, with depth inflated toward whoever was looking. Leaving
        it axis-aligned in the lab stands it up like a printed card, because its
        flat side then faces an arbitrary direction. Pointing its local +Z at
        the lab camera reproduces exactly the view the photo-crop was taken
        from, which is the only orientation the data actually supports.

        This applies ONLY to reconstructed meshes. The fallback primitive has no
        view-aligned relief to preserve, and turning a box to face the camera
        presents exactly one face - which is how a stand-in box ends up looking
        flatter than the reconstruction it stands in for.
        """
        f = _view_direction()
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), f)
        n = float(np.linalg.norm(right))
        if n < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / n
        up = np.cross(f, right)
        R = np.stack([right, up, f], axis=1).astype(np.float32)

        return self._centre_mesh(mesh.transformed(rotation=R))

    @staticmethod
    def _centre_mesh(mesh: Mesh) -> Mesh:
        """Centre a mesh on its own bounding box, in place.

        The per-step translate treats the object as straddling its centre, so
        anything that moved the geometry (a rotation, or the reconstruction's
        own mask centroid) has to be normalised out first.
        """
        v = mesh.vertices
        for axis in range(3):
            v[:, axis] -= 0.5 * (float(v[:, axis].min()) + float(v[:, axis].max()))
        return mesh

    def _ensure_static_background(self) -> None:
        """Rasterize AND shade the static lab once, caching both for the demo.

        Caching the finished image, not just the G-buffer, is what makes the
        per-frame cost proportional to the moving actors: each frame starts from
        this bake and re-shades only the pixels the object, its shadow, and the
        hand actually touch.
        """
        key = (tuple(np.round(self.camera.position, 4)),
               tuple(np.round(self.camera.target, 4)),
               round(self.camera.fov_y_deg, 3), self.width, self.height)
        if self._static_camera_key == key and self._static_snapshot is not None:
            return
        self._rast.clear()
        for mesh in self.scene.meshes:
            self._rast.draw(mesh, self.camera)
        self._static_snapshot = self._rast.snapshot()
        self._static_depth = self._rast.depth.copy()
        self._static_lit = SH.tonemap(
            SH.shade(self._rast, self.camera, self._lights, self._env)
            * self._vignette[..., None])
        self._static_camera_key = key

    # --------------------------------------------------------------- render

    def step_for_progress(self, progress: float) -> int:
        """Map the workflow's 0..1 phase progress onto a trajectory index.

        The plan is 2 s of motion but the demo phase holds for ~6 s, so it plays
        out over the first 82% and then holds on the final grasp - the held beat
        is what makes the result readable rather than a blink-and-miss loop.
        """
        n = len(self.trajectory.waypoints) if self.trajectory else 1
        play = float(np.clip(progress / 0.82, 0.0, 1.0))
        eased = play * play * (3.0 - 2.0 * play)
        return int(np.clip(round(eased * (n - 1)), 0, n - 1))

    def hand_screen_height(self, step: int) -> float:
        """On-screen pixel height of the hand at `step` - the framing check that
        matters, since a hand under ~30% of frame height stops reading as one."""
        if not self.is_ready:
            return 0.0
        screen, _ = self.camera.project(self._hand_paths_lab[int(step)])
        return float(screen[:, 1].max() - screen[:, 1].min())

    def telemetry(self, step: int) -> Dict[str, float]:
        """Per-step numbers for the caller's HUD - all from the real plan."""
        if not self.trajectory or not self.trajectory.waypoints:
            return {}
        n = len(self.trajectory.waypoints)
        wp = self.trajectory.waypoints[int(np.clip(step, 0, n - 1))]
        return {
            "step": int(step) + 1,
            "num_steps": n,
            "sim_time": float(wp.time_offset),
            "gripper": float(wp.gripper_aperture),
            "contact": float(np.mean(wp.contact_state)),
            "lift_cm": float(max(0.0, self._object_path_lab[step, 1]
                                 - self._object_path_lab[0, 1]) * 100.0),
        }

    def _hand_mesh_for(self, step: int) -> Mesh:
        return HM.build_hand_mesh(
            self._hand_paths_lab[step],
            color=_HOLOGRAM_COLOR,
            material=_HOLOGRAM_MATERIAL,
        )

    def _object_mesh_for(self, step: int) -> Mesh:
        return self.object_mesh.transformed(translation=self._object_path_lab[step])

    def _planar_shadow_mask(self, meshes: List[Mesh]) -> Optional[np.ndarray]:
        """Key-light shadow of the actors cast onto the staging surface.

        A true shadow map would need a second depth pass per light; a planar
        projection is exact for the one surface that matters here, and costs a
        single fillPoly per caster.

        Each caster gets its OWN blur radius from its own height above the
        plane. Averaging the heights instead - hand and object share one sigma -
        smears the object's contact shadow to the softness of the hand hovering
        8 cm up, and a contact shadow that soft stops anchoring anything.
        """
        light_dir = self._lights[0].direction                 # direction of travel
        if abs(float(light_dir[1])) < 1e-3:
            return None
        plane_y = LS.PEDESTAL_TOP_Y + 0.001

        combined = None
        for mesh in meshes:
            v = mesh.vertices
            above = v[:, 1] > plane_y + 1e-4
            if not above.any():
                continue
            t = (plane_y - v[:, 1]) / light_dir[1]
            projected = v + light_dir[None, :] * t[:, None]
            projected[:, 1] = plane_y
            screen, depth = self.camera.project(projected)
            visible = np.isfinite(depth) & (depth > self.camera.near)
            tri = mesh.faces
            # ALL three vertices must be above the plane. A triangle straddling
            # it projects its below-plane vertices in the opposite direction,
            # firing long spikes across the surface.
            ok = above[tri].all(axis=1) & visible[tri].all(axis=1)
            if not ok.any():
                continue

            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, list(np.rint(screen[tri[ok]]).astype(np.int32)), 255)
            height_m = float(np.mean(v[above, 1] - plane_y))
            sigma = float(np.clip(1.2 + height_m * 42.0, 1.2, 9.0))
            soft = cv2.GaussianBlur(mask.astype(np.float32) * (1.0 / 255.0), (0, 0), sigma)
            # Blurring a hard mask loses peak density; renormalise so a caster
            # sitting right on the surface still reads as a solid contact patch.
            peak = float(soft.max())
            if peak > 1e-3:
                soft *= min(1.0 / peak, 2.5)
            combined = soft if combined is None else np.maximum(combined, soft)

        return None if combined is None else np.clip(combined, 0.0, 1.0)

    def _bench_top_selector(self) -> np.ndarray:
        """Pixels belonging to the upward-facing worktop, where shadows land."""
        gb = self._rast.gbuffer
        world_y = gb[:, :, 6 + 1]
        normal_y = gb[:, :, 3 + 1]
        covered = gb[:, :, 12] > 0.5
        return covered & (normal_y > 0.45) & (world_y > LS.BENCH_TOP_Y - 0.04) \
            & (world_y < LS.PEDESTAL_TOP_Y + 0.012)

    def render(self, step: int, elapsed: float = 0.0,
               push_in: float = 0.0) -> Optional[np.ndarray]:
        """Render one frame of the reenactment as a BGR uint8 image.

        ``push_in`` in [0, 1] applies a slow zoom. A uniform scale about the
        principal point is mathematically identical to increasing the focal
        length, so this is a real camera move done as a 2-D crop - which keeps
        the static background bake valid for the whole demo.
        """
        if not self.is_ready:
            return None
        t0 = time.perf_counter()

        step = int(np.clip(step, 0, len(self.trajectory.waypoints) - 1))
        key = (step, round(push_in, 3))
        if self._last_image is not None:
            if key == self._last_key:
                # Identical content. Nothing to recompute, whatever the clock
                # says - gating this on the interval as well meant that once a
                # render took longer than the interval itself (which it does as
                # soon as the machine is loaded) the cache never engaged, and
                # the same frame was redrawn from scratch every call.
                return self._last_image
            if t0 - self._last_render_t < self._min_interval:
                # New content, but too soon to spend the client's frame budget.
                return self._last_image
        hand = self._hand_mesh_for(step)
        obj = self._object_mesh_for(step)

        self._rast.restore(self._static_snapshot)
        self._rast.draw(obj, self.camera)

        img = self._static_lit.copy()

        # Only the object's own pixels and the pixels its/the hand's shadow
        # darkens differ from the bake; everything else is already correct.
        dirty = self._rast.depth != self._static_depth

        shadow = self._planar_shadow_mask([obj, hand])
        shadow_rows = None
        if shadow is not None:
            lit_by_shadow = (shadow > 0.004) & self._bench_top_selector()
            dirty = dirty | lit_by_shadow

        idx = np.flatnonzero(dirty.ravel())
        if len(idx) > 0:
            gb_rows = self._rast.gbuffer.reshape(-1, 13)[idx]
            depth_rows = self._rast.depth.reshape(-1)[idx]
            if shadow is not None:
                atten = np.ones(len(idx), dtype=np.float32)
                on_bench = lit_by_shadow.ravel()[idx]
                atten[on_bench] = 1.0 - 0.82 * shadow.ravel()[idx][on_bench]
                shadow_rows = atten
            lit_rows = SH.shade_rows(gb_rows, depth_rows, self.camera,
                                     self._lights, self._env, shadow_rows)
            lit_rows *= self._vignette.reshape(-1)[idx][:, None]
            img.reshape(-1, 3)[idx] = SH.tonemap(lit_rows)

        # Hologram pass: the ghost hand is translucent, so it renders into its
        # own layer seeded with the opaque depth (bench and object still occlude
        # it correctly) and is composited with a fresnel-weighted alpha.
        self._ghost.clear()
        self._ghost.seed_depth(self._rast.depth)
        self._ghost.draw(hand, self.camera)
        self._composite_hologram(img, elapsed)

        if push_in > 1e-3:
            img = self._apply_push_in(img, push_in)

        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        self._last_render_t = t0
        self._last_key = key
        self._last_image = img
        return img

    def _composite_hologram(self, img: np.ndarray, elapsed: float) -> None:
        """Alpha-blend the ghost hand into `img` in place."""
        gb = self._ghost.gbuffer
        idx = np.flatnonzero((gb[:, :, 12] > 0.5).ravel())
        if len(idx) == 0:
            return

        gb_rows = gb.reshape(-1, 13)[idx]
        ghost_lit = SH.shade_rows(gb_rows, self._ghost.depth.reshape(-1)[idx],
                                  self.camera, self._lights, self._env)
        ghost_lit *= self._vignette.reshape(-1)[idx][:, None]
        ghost_u8 = SH.tonemap(ghost_lit).astype(np.float32)

        normal = gb_rows[:, 3:6]
        world = gb_rows[:, 6:9]
        n = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        v = self.camera.position[None, :] - world
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        facing = np.abs(np.sum(n * v, axis=1))
        # Fresnel: grazing angles go opaque, so the silhouette reads as a solid
        # edge while the interior stays see-through - the classic hologram cue.
        alpha = 0.34 + 0.60 * (1.0 - facing) ** 2.0

        rows_y = (idx // self.width).astype(np.float32)
        alpha *= 0.90 + 0.10 * np.sin(rows_y * 0.55 - elapsed * 7.0)
        alpha = np.clip(alpha, 0.0, 0.97)[:, None]

        flat = img.reshape(-1, 3)
        flat[idx] = (flat[idx].astype(np.float32) * (1.0 - alpha)
                     + ghost_u8 * alpha).astype(np.uint8)

    def _apply_push_in(self, img: np.ndarray, amount: float) -> np.ndarray:
        zoom = 1.0 + 0.07 * float(np.clip(amount, 0.0, 1.0))
        h, w = img.shape[:2]
        cw, ch = int(round(w / zoom)), int(round(h / zoom))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
