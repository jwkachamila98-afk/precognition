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
from src.simulation.render import object_library as OL
from src.simulation.render import object_mesh as OM
from src.simulation.render import shading as SH
from src.simulation.render.camera import Camera
from src.simulation.render.raster import Material, Mesh, Rasterizer
from src.simulation.render.scene_mesh import build_scene_mesh
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

# Where the hand comes from, as an azimuth OFFSET from the camera. The plan's
# own grasp azimuth is arbitrary - a canned orientation, not a measurement - so
# against a locked camera it is as likely to be unreadable as not.
#
# Chosen by measuring, because the two obvious criteria disagree. Aligning the
# palm to the camera peaks at 150-180 degrees, but that puts the hand BEHIND the
# object, where it is occluded down to a sliver: best-facing scored the worst
# visible area of any angle tested (0.55% of frame against 2.12%). Meanwhile the
# 40-80 degree band turns the palm edge-on, rendering the hand as a featureless
# slab. What actually matters is hand pixels that survive the depth test, which
# peaks near 330 degrees - the viewer's side of the object, off-axis enough to
# show the grasp closing, with the palm well clear of edge-on.
_HAND_APPROACH_OFFSET_DEG = 330.0

# How long the completed grasp is held on screen once the plan has played out.
_END_HOLD_SEC = 1.4
# Fewest tracked frames that can carry a reach. Below roughly two seconds of
# tracking there is no motion to show, only a pose.
_MIN_DEMONSTRATION_POSES = 24

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
_CAM_FOV_Y_DEG = 42.0
# The camera now follows the real viewpoint (see _fit_camera), so distance is
# derived from the object rather than fixed, and elevation is floored so a
# webcam level with the object does not put the lab camera inside the bench.
_MIN_CAMERA_ELEVATION_DEG = 14.0
_FRAMING_SPAN_MULTIPLE = 3.4
_CAM_TARGET_LIFT_M = 0.11


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
# A hand manipulates plenty of things longer than itself - you take a 25 cm
# bottle by the neck - so the ceiling is generous. Its job is to reject the
# absurd (56 cm coffee cups, 74 cm wine glasses), not to enforce daintiness.
_MIN_OBJECT_PALMS = 0.45
_MAX_OBJECT_PALMS = 3.2

# Typical longest dimension, in metres, for the things people actually pick up.
# These are priors on the OBJECT CLASS, and they beat the detector's own extent
# outright, because that extent is back-projected through depth which is
# synthetic locally and only relative (MiDaS) on GPU - non-metric either way. It
# reported a wine glass at 74 cm on a GPU pod and a coffee cup at 56 cm here. A
# prior cannot tell a large mug from a small one, but it is never wrong by 6x.
_CLASS_SIZE_PRIORS_M = {
    "cup": 0.09, "mug": 0.10, "wine glass": 0.20, "bowl": 0.15, "bottle": 0.25,
    "can": 0.12, "fork": 0.19, "knife": 0.22, "spoon": 0.17, "banana": 0.19,
    "apple": 0.08, "orange": 0.08, "remote": 0.16, "cell phone": 0.15,
    "phone": 0.15, "mouse": 0.11, "book": 0.24, "scissors": 0.16, "pen": 0.14,
    "stylus": 0.14, "toothbrush": 0.19, "clock": 0.20, "vase": 0.25,
    "sports ball": 0.22, "teddy bear": 0.28, "hair drier": 0.25,
    # The weakest prior in the table by some distance - a "box" could be any
    # size at all - but a shoebox-ish 20 cm beats the 67 cm the detector's
    # back-projection produces for objects it cannot resolve.
    "box": 0.20,
}


def _size_prior_for(label: Optional[str]) -> Optional[float]:
    """Prior for a detector label, matched on the longest contained keyword.

    Longest-first so "wine glass" is not shadowed by a substring match, and
    "cell phone" resolves ahead of "phone".
    """
    if not label:
        return None
    text = label.replace("_", " ").strip().lower()
    for key in sorted(_CLASS_SIZE_PRIORS_M, key=len, reverse=True):
        if key in text:
            return _CLASS_SIZE_PRIORS_M[key]
    return None


def _object_longest_dimension(bbox: Optional[BoundingBox3D],
                              hand_palm_m: float,
                              label: Optional[str] = None) -> float:
    """The manipuland's longest dimension in metres, in a scale the plan supports.

    Preference order, weakest evidence last:
      1. a size prior for the detected CLASS, when the label names something
         known - see _CLASS_SIZE_PRIORS_M for why this outranks measurement;
      2. the detector's own 3-D extent, for anything unrecognised;
      3. a hand-relative default when there is no detection at all.

    Whatever the source, the result is bounded by what the grasping hand could
    actually close around. Only the SHAPE comes from the silhouette (see
    object_mesh.build_object_mesh); this is scale alone.
    """
    prior = _size_prior_for(label or (bbox.label if bbox is not None else None))
    raw = prior
    source = "class prior"

    if raw is None:
        source = "detector extent"
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
            f"LabSimulator: {source} put the target at {raw*100:.0f} cm, which the "
            f"grasping hand ({hand_palm_m*100:.1f} cm palm) could not handle; staging "
            f"it at {clamped*100:.0f} cm instead."
        )
    elif prior is not None:
        detected = (float(np.asarray(bbox.size, dtype=np.float32).max())
                    if bbox is not None else float("nan"))
        logger.info(
            f"LabSimulator: staging '{label or (bbox.label if bbox else '?')}' at its class "
            f"prior of {clamped*100:.0f} cm (the detector's non-metric extent said "
            f"{detected*100:.0f} cm)."
        )
    return clamped


class LabSimulator:
    """Renders the Autonomous Demo reenactment inside the simulated lab."""

    def __init__(self, width: int = 384, height: int = 288,
                 max_fps: float = 20.0, registered: bool = False) -> None:
        # REGISTERED mode drops the studio and puts the reenactment back in the
        # real scene: the camera is the webcam's own pinhole, the object keeps
        # the position and extent it was detected at, the hand keeps its tracked
        # metric position, and the background is the live frame. The object then
        # lands on exactly the pixels it occupies in the video - not because
        # anything is fitted to match, but because nothing is moved.
        #
        # The studio look is not deleted, only switched off: it frames every
        # reenactment identically, which registration cannot do.
        self.registered = bool(registered)
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
        # The real room, reconstructed from depth. When present the registered
        # render is entirely geometry - no video plate anywhere in the shot -
        # and, being static for the length of a demo, it is baked exactly like
        # the studio was: (lit image, coverage, g-buffer snapshot, depth).
        self.scene_mesh: Optional[Mesh] = None
        self._scene_bake = None
        self.transform: Optional[LabTransform] = None
        self.trajectory: Optional[ForeseenTrajectory] = None
        self.object_mesh: Optional[Mesh] = None
        self.object_size = np.array([0.13, 0.08, 0.05], dtype=np.float32)
        self.object_is_reconstructed = False
        self.target_label = "object"
        self._contact_override: Optional[int] = None
        self._is_demonstration = False
        self._timestamps: Optional[np.ndarray] = None
        # Step count of whatever is staged. NOT len(trajectory.waypoints): a
        # recorded demonstration has no trajectory, and reading through one
        # crashed the render loop the first time a demonstration was played.
        self._num_steps = 0
        self._hand_paths_lab: Optional[np.ndarray] = None      # (T, 21, 3)
        self._object_path_lab: Optional[np.ndarray] = None     # (T, 3)
        self._ready = False
        self.last_render_ms = 0.0

        # Set by the client from the workflow, so the playback window matches the
        # phase the server is actually holding.
        self.demo_duration_sec = 12.0
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

    def prepare_from_demonstration(
        self,
        recorded_poses: Optional[List[HandPose]],
        target_bbox: Optional[BoundingBox3D],
        sprite: Optional[np.ndarray],
    ) -> bool:
        """Stage the user's OWN recorded motion, carrying the object with it.

        The hand is the 21-joint pose actually tracked while they performed the
        grasp - not a plan, not a policy correction. The object stands still
        until the hand reaches it and then travels with it, which is the only
        object motion a hand recording can honestly support: a tracked hand
        carries no object physics of its own.

        Contact is taken as the frame where the fingertips come closest to the
        object in 3-D. From there the object inherits the hand's displacement
        since that frame, so it is carried rather than scripted.
        """
        if (not recorded_poses or len(recorded_poses) < _MIN_DEMONSTRATION_POSES
                or target_bbox is None):
            # Too short to be a replay of anything. A handful of frames is what
            # you get when the execution phase is entered and left almost
            # immediately, and staging it produced a "reenactment" seven frames
            # long that was over before it could be seen - while claiming, in
            # the log and on screen, to be showing the user their own motion.
            # Better to fall back to the generated plan and say so.
            if recorded_poses and target_bbox is not None:
                logger.info(
                    f"LabSimulator: only {len(recorded_poses)} tracked frames - too "
                    f"short to replay, falling back to the generated plan."
                )
            self._ready = False
            return False

        kpts = np.stack([p.keypoints_3d for p in recorded_poses]).astype(np.float32)
        centre = np.asarray(target_bbox.center, dtype=np.float32)

        tips = kpts[:, _FINGERTIPS].mean(axis=1)
        contact = int(np.argmin(np.linalg.norm(tips - centre[None, :], axis=1)))

        # Object: fixed at its detected position until contact, then carried.
        hand_ref = kpts[:, HM.WRIST]
        object_path_cam = np.tile(centre, (len(kpts), 1)).astype(np.float32)
        object_path_cam[contact:] += hand_ref[contact:] - hand_ref[contact]

        # A recording can run a minute or more; the demo window is twelve seconds.
        # Squeezing the whole thing in plays it at five times real speed, which
        # misrepresents the very thing the replay exists to show. Keep the part
        # that carries the grasp, at its own pace, and drop the rest.
        # float64 THEN rebase. These are absolute Unix timestamps around 1.79e9,
        # where float32 has a 128-second ULP: casting them directly collapsed a
        # 2050-frame recording to two distinct values, so the duration read as
        # zero, the trim below never fired, and real-speed playback interpolated
        # against a constant array. Relative seconds are small and fit fine.
        stamps = np.asarray([p.timestamp for p in recorded_poses], dtype=np.float64)
        stamps = (stamps - stamps[0]).astype(np.float32)
        if len(stamps) > 1 and float(stamps[-1] - stamps[0]) > 0.0:
            duration = float(stamps[-1] - stamps[0])
            budget = max(self.demo_duration_sec - _END_HOLD_SEC, 1.0)
            if duration > budget:
                # The whole reach is replayed, compressed to fit. It used to be
                # trimmed to a window around the grasp instead, on the grounds
                # that real speed matters more than completeness - but a replay
                # that silently drops the beginning of the movement is not a
                # replay of the movement, and the missing start is exactly where
                # the approach the policy is learning from happens.
                logger.info(
                    f"LabSimulator: recording runs {duration:.1f}s against a "
                    f"{budget:.1f}s window; replaying all of it at "
                    f"{duration / budget:.1f}x speed rather than cutting it."
                )

        self._contact_override = contact
        return self._stage(
            raw_kpts=kpts,
            object_path_cam=object_path_cam,
            label=target_bbox.label or "object",
            target_bbox=target_bbox,
            sprite=sprite,
            source="your recorded demonstration",
            timestamps=stamps,
        )

    def prepare(self, trajectory: Optional[ForeseenTrajectory],
                target_bbox: Optional[BoundingBox3D],
                sprite: Optional[np.ndarray]) -> bool:
        """Stage a generated plan in the lab. Call once when the demo starts.

        Returns False if there is nothing renderable, in which case the caller
        should fall back to the 2-D overlay.
        """
        if trajectory is None or not trajectory.waypoints:
            self._ready = False
            return False

        self.trajectory = trajectory
        self._contact_override = None
        wp = trajectory.waypoints
        return self._stage(
            raw_kpts=np.stack([w.hand_keypoints_3d for w in wp]).astype(np.float32),
            object_path_cam=np.stack([w.object_pose[:3] for w in wp]).astype(np.float32),
            label=(trajectory.target_label or
                   (target_bbox.label if target_bbox else "object")),
            target_bbox=target_bbox,
            sprite=sprite,
            source="a generated plan",
            timestamps=np.array([w.time_offset for w in wp], dtype=np.float32),
        )

    def _stage(self, raw_kpts: np.ndarray, object_path_cam: np.ndarray,
               label: str, target_bbox: Optional[BoundingBox3D],
               sprite: Optional[np.ndarray], source: str,
               timestamps: np.ndarray) -> bool:
        """Everything common to staging a plan and staging a demonstration."""
        t0 = time.perf_counter()
        self.target_label = label
        self._timestamps = timestamps
        self._is_demonstration = self._contact_override is not None

        # Hand scale first: it is the metric reference the object is sized against.
        measured_palm = float(np.median(
            np.linalg.norm(raw_kpts[:, HM._MCP["middle"]] - raw_kpts[:, HM.WRIST], axis=1)))
        if self.registered:
            # Nothing is normalised: rescaling the hand or substituting a class
            # prior for the object's extent would move both off the pixels they
            # occupy in the video, which is the whole point of this mode.
            #
            # The class prior is the better estimate of how big the thing really
            # IS - the detector's extent is back-projected through depth that is
            # synthetic locally and only relative on GPU, so it has been wrong by
            # 6x. But registration does not want metric truth, it wants agreement
            # with the observation: the extent and the centre come from the same
            # back-projection, so drawing the object at both reproduces the pixels
            # it was measured from, however wrong the metres are.
            scale = 1.0
            longest = (float(np.max(np.asarray(target_bbox.size, dtype=np.float32)))
                       if target_bbox is not None else _REFERENCE_PALM_M)
        else:
            scale = float(np.clip(_REFERENCE_PALM_M / max(measured_palm, 1e-4), 0.15, 8.0))
            longest = _object_longest_dimension(target_bbox, _REFERENCE_PALM_M, label=label)

        origin_cam = (np.asarray(target_bbox.center, dtype=np.float32)
                      if target_bbox is not None else object_path_cam[0].copy())

        # Shape, in descending order of how much is actually known:
        #   1. a profile authored for THIS object, when one is available;
        #   2. the class's canonical geometry;
        #   3. silhouette inflation from the photo-crop;
        #   4. a plain box.
        mesh = None
        self.object_is_reconstructed = False
        try:
            mesh = OL.build_class_mesh(label, longest, sprite,
                                       profile_cm=OL.authored_profile(label))
        except Exception as exc:
            logger.warning(f"LabSimulator: class mesh failed ({exc}); trying the silhouette.")

        if mesh is None and sprite is not None and sprite.size > 0:
            try:
                mesh = OM.build_object_mesh(sprite, longest, contour_points=24, rings=2)
                self.object_is_reconstructed = mesh is not None
            except Exception as exc:                      # reconstruction is best-effort
                logger.warning(f"LabSimulator: object reconstruction failed ({exc}); using primitive.")
        if mesh is None:
            mesh = OM.fallback_object_mesh(longest)

        mesh = self._orient_to_camera(mesh) if self.object_is_reconstructed \
            else self._centre_mesh(mesh)
        self.object_mesh = mesh
        self.object_size = OM.mesh_extent(mesh)

        if self.registered:
            # The identity: perception metres, axis-flipped into the lab's
            # handedness and left exactly where they were measured.
            anchor_lab = np.zeros(3, dtype=np.float32)
            self.transform = LabTransform(origin_cam=np.zeros(3, dtype=np.float32),
                                          anchor_lab=anchor_lab, scale=1.0)
        else:
            anchor_lab = LS.OBJECT_ANCHOR + np.array(
                [0.0, float(self.object_size[1]) * 0.5, 0.0], dtype=np.float32)
            self.transform = LabTransform(origin_cam=origin_cam, anchor_lab=anchor_lab,
                                          scale=scale)

        self._num_steps = int(len(raw_kpts))
        self._hand_paths_lab = self.transform(raw_kpts.reshape(-1, 3)).reshape(raw_kpts.shape)
        self._object_path_lab = self.transform(object_path_cam)
        if not self.registered:
            # Keep the staged object from sinking through or floating above the
            # bench: its own motion is preserved, its resting height corrected.
            # Registered mode has no bench to rest on, and shifting the object
            # vertically would slide it off the pixels it was detected at.
            self._object_path_lab[:, 1] += (float(anchor_lab[1])
                                            - float(self._object_path_lab[0, 1]))

        # A DEMONSTRATION is left exactly as recorded. Rotating it to suit the
        # camera, or sliding the grasp onto the staged object, would be editing
        # the very thing it exists to show. A generated plan carries no such
        # claim - its orientation is canned - so it is still posed for legibility.
        if not self._is_demonstration and not self.registered:
            self._present_hand_to_the_camera()
            self._seat_grasp_on_the_staged_object()

        self.camera = self._registered_camera() if self.registered else self._fit_camera()
        if not self.registered:
            self._ensure_static_background()
        self._ready = True

        logger.info(
            f"LabSimulator: staged '{self.target_label}' from {source} "
            f"({self.object_size[0]*100:.0f}x{self.object_size[1]*100:.0f}x"
            f"{self.object_size[2]*100:.0f} cm, hand scale x{scale:.2f}) across "
            f"{len(raw_kpts)} steps - lab {self.scene.triangle_count} tris static, "
            f"{self.object_mesh.num_faces} object tris, prepared in "
            f"{(time.perf_counter() - t0)*1000:.0f} ms."
        )
        return True

    def _present_hand_to_the_camera(self) -> None:
        """Rotate a GENERATED plan's hand path about the object's vertical axis so
        the grasp is seen at a three-quarter view rather than edge-on.

        A rigid rotation about the axis the object stands on: the hand's motion
        relative to the object, its distance, and the grasp itself are all
        unchanged - only which side it comes from. That is not information a
        canned plan actually carries. It is NOT applied to a recorded
        demonstration, where the approach direction is real.

        Chosen by measurement: aligning the palm to the camera peaks at 150-180
        degrees, but that puts the hand behind the object where it occludes down
        to a sliver (0.55% of frame against 2.12%); 40-80 degrees turns the palm
        edge-on. Visible pixels after depth testing peak near 330 degrees.
        """
        grasp = self._contact_step()
        obj = self._object_path_lab[grasp]
        wrist = self._hand_paths_lab[grasp][HM.WRIST]

        offset = wrist - obj
        if abs(offset[0]) < 1e-4 and abs(offset[2]) < 1e-4:
            return                                    # directly overhead; no azimuth
        current = math.atan2(float(offset[0]), float(offset[2]))
        turn = math.radians(_CAM_AZ_DEG + _HAND_APPROACH_OFFSET_DEG) - current

        c, s_ = math.cos(turn), math.sin(turn)
        R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]], dtype=np.float32)
        pivot = np.array([obj[0], 0.0, obj[2]], dtype=np.float32)
        flat = self._hand_paths_lab.reshape(-1, 3) - pivot
        self._hand_paths_lab = (flat @ R.T + pivot).reshape(self._hand_paths_lab.shape)

    def _seat_grasp_on_the_staged_object(self) -> None:
        """Re-seat a GENERATED plan's hand so its grasp lands on the object as
        staged.

        The planner sizes its approach from the detector's raw extent, but the
        lab stages the object at a corrected scale, and when those disagree the
        hand closes in the air above a smaller object. Shifting the whole path
        by a constant preserves the plan's relative motion and the object's lift.

        Not applied to a recorded demonstration: there the hand's position
        relative to the object is measured, and overriding it would edit the
        very thing the replay exists to show.
        """
        grasp = self._contact_step()
        tips = self._hand_paths_lab[grasp][_FINGERTIPS]
        pinch = 0.5 * (tips[0] + tips[[1, 2]].mean(axis=0))
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

    def set_scene_from_depth(self, depth_m: Optional[np.ndarray],
                             frame_bgr: Optional[np.ndarray],
                             grid_w: int = 72) -> bool:
        """Reconstruct the room from a depth map and bake it for this demo.

        Called once when the demo is staged, not per frame: the camera does not
        move and neither does the room, so re-deriving either would be paying
        several thousand triangles a frame for an identical picture.
        """
        self.scene_mesh = None
        self._scene_bake = None
        if not self.registered or depth_m is None or frame_bgr is None:
            return False
        mesh = build_scene_mesh(depth_m, frame_bgr, grid_w=grid_w)
        if mesh is None or self.camera is None:
            return False

        t0 = time.perf_counter()
        self._rast.clear()
        self._rast.draw(mesh, self.camera)
        snapshot = self._rast.snapshot()
        depth = self._rast.depth.copy()
        lit = SH.tonemap(SH.shade(self._rast, self.camera, self._lights, self._env))
        cover = (self._rast.gbuffer[:, :, 12] > 0.5)[:, :, None]

        self.scene_mesh = mesh
        self._scene_bake = (lit, cover, snapshot, depth)
        logger.info(
            f"LabSimulator: reconstructed the scene as {mesh.num_faces} triangles "
            f"covering {100.0 * cover.mean():.0f}% of frame, baked in "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms."
        )
        return True

    def _registered_camera(self) -> Camera:
        """The webcam itself, as a lab camera.

        The perception frame puts the camera at the origin looking along +Z with
        fx = fy = 0.8 * width and the principal point at the image centre. In lab
        world that is the origin looking along -Z, and the only thing left to
        derive is the vertical field of view:

            tan(fov_y / 2) = (h / 2) / (0.8 * w)

        which is resolution-independent at a fixed aspect - 50.23 degrees for
        4:3, against the 42 the studio rig used. With this camera the projection
        agrees with BoundingBox3D.project_to_2d to the pixel, so an object drawn
        at its detected position lands on the pixels it was detected from.
        """
        fov_y = math.degrees(2.0 * math.atan(
            (self.height * 0.5) / (0.8 * float(self.width))))
        return Camera(position=np.zeros(3, dtype=np.float32),
                      target=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                      up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                      fov_y_deg=fov_y, width=self.width, height=self.height)

    def _fit_camera(self) -> Camera:
        """Look at the staged object from the direction the real camera saw it.

        The lab camera is aimed along the vector from the object to the origin
        of the PERCEPTION frame - which is where the webcam is - so the sim view
        lines up with what the user is actually looking at. An object seen from
        above and to the left is staged as seen from above and to the left.

        This deliberately replaces the fixed pose used earlier. The trade is the
        one that came with it: framing now changes as the user moves, where
        before every demo was filmed identically. Distance is still governed
        here rather than copied, since a webcam at arm's length crops the bench
        out entirely, and the elevation is floored so the camera cannot end up
        underneath the worktop looking up through it.
        """
        anchor = self.transform.anchor_lab

        # Object -> camera, in the perception frame, mapped into the lab.
        to_camera_cam = -np.asarray(self.transform.origin_cam, dtype=np.float32)
        direction = (_AXIS_FLIP @ to_camera_cam).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        direction = (direction / norm) if norm > 1e-6 else _view_direction()

        # Floor the elevation: a real camera is often level with the object, but
        # a lab camera at bench height sees the worktop edge-on and little else.
        min_el = math.radians(_MIN_CAMERA_ELEVATION_DEG)
        horiz = float(np.hypot(direction[0], direction[2]))
        if horiz < 1e-6:
            direction = _view_direction()
        elif math.atan2(float(direction[1]), horiz) < min_el:
            scale_h = math.cos(min_el) / horiz
            direction = np.array([direction[0] * scale_h, math.sin(min_el),
                                  direction[2] * scale_h], dtype=np.float32)
        # Never end up behind the backdrop.
        if direction[2] <= 0.05:
            direction[2] = 0.05
            direction /= max(float(np.linalg.norm(direction)), 1e-6)

        span = max(float(self.object_size.max()), 0.05)
        dist = float(np.clip(span * _FRAMING_SPAN_MULTIPLE, 0.45, 1.60))

        target = anchor + np.array([0.0, _CAM_TARGET_LIFT_M, 0.0], dtype=np.float32)
        position = target + direction * dist
        position[1] = float(max(position[1], LS.BENCH_TOP_Y + 0.18))
        position[2] = float(max(position[2], LS.BENCH_FRONT_Z + 0.08))

        return Camera(position=position, target=target, up=(0.0, 1.0, 0.0),
                      fov_y_deg=_CAM_FOV_Y_DEG, width=self.width, height=self.height,
                      near=0.04, far=40.0)

    def _contact_step(self) -> int:
        """The step where the grasp closes - the hero beat.

        For a recorded demonstration this was measured from the fingertips'
        closest approach; a plan declares it in its own contact_state.
        """
        if self._contact_override is not None:
            return int(self._contact_override)
        if self.trajectory is None:
            return max(self._num_steps // 2, 0)
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

    def step_for_progress(self, progress: float) -> float:
        """Map the workflow's 0..1 phase progress onto a trajectory position.

        The plan is 2 s of motion but the demo phase holds for ~6 s, so it plays
        out over the first 82% and then holds on the final grasp - the held beat
        is what makes the result readable rather than a blink-and-miss loop.

        Returns a FRACTIONAL index. Rounding it to a whole waypoint quantised the
        motion to 60 discrete poses across the demo, which reads as a stutter
        however smoothly the progress itself advances; callers interpolate
        between the neighbouring waypoints instead.
        """
        n = max(self._num_steps, 1)

        if self._is_demonstration and self._timestamps is not None and n > 1:
            # Real speed where it fits, compressed where it does not - but the
            # whole recording plays either way. No easing: the recording carries
            # its own acceleration and smoothing it again would be editing it.
            #
            # Mapping wall-clock straight onto recorded time is real speed, and
            # that is right until the reach outlasts the window: then playback
            # simply stopped wherever the phase ran out, silently dropping the
            # end of the movement - including the grasp, if the approach was
            # slow. A replay that omits the grasp is not showing the attempt.
            span = max(self.demo_duration_sec, 0.5)
            play_until = float(np.clip(1.0 - _END_HOLD_SEC / span, 0.5, 0.97))
            recorded = float(self._timestamps[-1] - self._timestamps[0])
            playable = span * play_until
            if recorded <= playable:
                elapsed = float(np.clip(progress, 0.0, 1.0)) * span
            else:
                play = float(np.clip(progress / play_until, 0.0, 1.0))
                elapsed = play * recorded
            return float(np.clip(
                np.interp(self._timestamps[0] + elapsed, self._timestamps, np.arange(n)),
                0.0, n - 1))

        # Hold the finished grasp for a FIXED beat rather than a fixed fraction:
        # a proportion that reads as a pause at six seconds reads as dead air at
        # twelve.
        span = max(self.demo_duration_sec, 0.5)
        play_until = float(np.clip(1.0 - _END_HOLD_SEC / span, 0.5, 0.97))
        play = float(np.clip(progress / play_until, 0.0, 1.0))
        eased = play * play * (3.0 - 2.0 * play)
        return float(np.clip(eased * (n - 1), 0.0, n - 1))

    @staticmethod
    def _lerp_along(path: np.ndarray, frame: float) -> np.ndarray:
        """Sample a per-waypoint array at a fractional index."""
        n = len(path)
        t = float(np.clip(frame, 0.0, n - 1))
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, n - 1)
        f = t - i0
        if f <= 1e-6:
            return path[i0]
        return (path[i0] * (1.0 - f) + path[i1] * f).astype(np.float32)

    def hand_screen_height(self, step: int) -> float:
        """On-screen pixel height of the hand at `step` - the framing check that
        matters, since a hand under ~30% of frame height stops reading as one."""
        if not self.is_ready:
            return 0.0
        screen, _ = self.camera.project(self._hand_paths_lab[int(step)])
        return float(screen[:, 1].max() - screen[:, 1].min())

    def visible_hand_fraction(self, step: float) -> float:
        """Fraction of the viewport the hand actually occupies AFTER occlusion.

        The readability metric that matters. Screen height is a poor proxy: an
        approach angled toward the viewer is shorter in screen-Y while showing
        considerably more of itself, and an angle that maximises palm alignment
        can put the hand behind the object where almost none of it survives the
        depth test at all.
        """
        if not self.is_ready:
            return 0.0
        self._rast.restore(self._static_snapshot)
        self._rast.draw(self._object_mesh_for(step), self.camera)
        self._ghost.clear()
        self._ghost.seed_depth(self._rast.depth)
        self._ghost.draw(self._hand_mesh_for(step), self.camera)
        visible = int((self._ghost.gbuffer[:, :, 12] > 0.5).sum())
        return visible / float(self.width * self.height)

    def telemetry(self, step: float) -> Dict[str, float]:
        """Per-step numbers for the caller's HUD.

        A generated plan states its own gripper aperture and per-fingertip
        contact. A recorded demonstration states neither - a hand tracker
        reports joints, not grip force - so those are DERIVED from the geometry:
        aperture from how far the fingertips have closed relative to their
        widest spread in the recording, contact from proximity to the object.
        Derived, and labelled as such here, rather than presented as measured.
        """
        if self._num_steps == 0:
            return {}
        n = self._num_steps
        idx = int(np.clip(round(step), 0, n - 1))

        lifted = float(self._lerp_along(self._object_path_lab, step)[1])
        common = {
            "step": idx + 1,
            "num_steps": n,
            "lift_cm": float(max(0.0, lifted - self._object_path_lab[0, 1]) * 100.0),
        }

        if self.trajectory is not None:
            wp = self.trajectory.waypoints[idx]
            grip = float(np.interp(step, np.arange(n),
                                   [w.gripper_aperture for w in self.trajectory.waypoints]))
            common.update(sim_time=float(wp.time_offset), gripper=grip,
                          contact=float(np.mean(wp.contact_state)))
            return common

        tips = self._hand_paths_lab[:, _FINGERTIPS]
        spread = np.linalg.norm(tips.max(axis=1) - tips.min(axis=1), axis=1)
        widest = float(spread.max())
        here = float(self._lerp_along(spread[:, None], step)[0])
        grip = float(np.clip(1.0 - here / max(widest, 1e-6), 0.0, 1.0))

        reach = float(np.linalg.norm(
            self._lerp_along(tips.mean(axis=1), step)
            - self._lerp_along(self._object_path_lab, step)))
        contact = float(np.clip(1.0 - reach / max(float(self.object_size.max()), 1e-3),
                                0.0, 1.0))
        when = (float(self._lerp_along(self._timestamps[:, None], step)[0]
                      - self._timestamps[0])
                if self._timestamps is not None and len(self._timestamps) == n else 0.0)
        common.update(sim_time=when, gripper=grip, contact=contact)
        return common

    def _hand_mesh_for(self, step: float) -> Mesh:
        return HM.build_hand_mesh(
            self._lerp_along(self._hand_paths_lab, step),
            color=_HOLOGRAM_COLOR,
            material=_HOLOGRAM_MATERIAL,
        )

    def _object_mesh_for(self, step: float) -> Mesh:
        return self.object_mesh.transformed(
            translation=self._lerp_along(self._object_path_lab, step))

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

    def _render_registered(self, step: float, elapsed: float,
                           background: np.ndarray) -> np.ndarray:
        """Draw the object and hand over the live frame, in the real camera.

        There is no static bake here and nothing to cache: the background is a
        different image every frame.

        With a `scene_mesh` set, the room is real geometry too and the frame is
        only what shows through the holes the reconstruction could not fill -
        where depth straddled a silhouette and the cell was dropped rather than
        draped across the gap. Leaving the photograph visible there is a more
        honest seam than inventing a surface over it.
        """
        img = background
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img.copy())

        obj = self._object_mesh_for(step)
        if self._scene_bake is not None:
            # The room is static geometry for the whole demo, so it is baked
            # once and only the actors are redrawn - the same trick the studio
            # used, and the reason this costs about what the studio did rather
            # than rasterising several thousand background triangles per frame.
            lit_bake, cover, snapshot, base_depth = self._scene_bake
            np.copyto(img, lit_bake, where=cover)
            self._rast.restore(snapshot)
            self._rast.draw(obj, self.camera)
            dirty = self._rast.depth != base_depth
            idx = np.flatnonzero(dirty.ravel())
        else:
            self._rast.clear()
            self._rast.draw(obj, self.camera)
            idx = np.flatnonzero((self._rast.gbuffer[:, :, 12] > 0.5).ravel())

        if len(idx) > 0:
            gb_rows = self._rast.gbuffer.reshape(-1, 13)[idx]
            lit = SH.shade_rows(gb_rows, self._rast.depth.reshape(-1)[idx],
                                self.camera, self._lights, self._env)
            img.reshape(-1, 3)[idx] = SH.tonemap(lit)

        # The hand is seeded with the object's depth so the object still occludes
        # it. Nothing else can: the real scene has no geometry here, so a hand
        # behind a real table edge will draw over it. That is the honest limit of
        # compositing without scene reconstruction.
        self._ghost.clear()
        self._ghost.seed_depth(self._rast.depth)
        self._ghost.draw(self._hand_mesh_for(step), self.camera)
        self._composite_hologram(img, elapsed)
        return img

    def render(self, step: float, elapsed: float = 0.0,
               push_in: float = 0.0,
               background: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Render one frame of the reenactment as a BGR uint8 image.

        ``push_in`` in [0, 1] applies a slow zoom. A uniform scale about the
        principal point is mathematically identical to increasing the focal
        length, so this is a real camera move done as a 2-D crop - which keeps
        the static background bake valid for the whole demo.
        """
        if not self.is_ready:
            return None
        t0 = time.perf_counter()

        step = float(np.clip(step, 0.0, max(self._num_steps - 1, 0)))

        if self.registered:
            if background is None:
                return None          # nothing to register against
            # Deliberately uncached: the background moves every frame, so a hit
            # on (step, push_in) would freeze the live image behind the actors.
            img = self._render_registered(step, elapsed, background)
            self.last_render_ms = (time.perf_counter() - t0) * 1000.0
            self._last_render_t = t0
            self._last_image = img
            return img

        key = (round(step, 3), round(push_in, 3))
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
