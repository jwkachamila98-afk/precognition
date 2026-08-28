"""Tests for the simulated-lab software renderer.

These cover the invariants that are expensive to eyeball and easy to break
silently - most of which DID break during development:

  * face winding (a mesh rendering its own inside is nearly invisible in a
    thumbnail but wrecks depth sorting and shadows);
  * the perception <-> lab coordinate conversion (a sign error here puts the
    hand under the bench);
  * object seating and framing (the object must rest on the stage, and no part
    of the plan may leave the viewport).
"""

import numpy as np
import pytest

from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.perception.scene_parser import BoundingBox3D
from src.simulation.lab_sim import LabSimulator, _view_direction
from src.simulation.render import hand_mesh as HM
from src.simulation.render import lab_scene as LS
from src.simulation.render import object_mesh as OM
from src.simulation.render import primitives as P
from src.simulation.render import shading as SH
from src.simulation.render.camera import Camera
from src.simulation.render.raster import Material, Rasterizer, orient_faces_outward


def _nearest_depth(mesh, eye, target=(0, 0, 0), up=(0, 1, 0), n=192):
    cam = Camera(position=eye, target=target, up=up, fov_y_deg=30, width=n, height=n)
    rast = Rasterizer(n, n)
    rast.draw(mesh, cam)
    finite = rast.depth[np.isfinite(rast.depth)]
    return float(finite.min()) if len(finite) else float("nan")


@pytest.fixture(scope="module")
def staged_sim():
    """A LabSimulator staged with a plan for a plausible object."""
    bbox = BoundingBox3D(
        label="wine glass",
        center=np.array([0.02, 0.05, 0.46], dtype=np.float32),
        size=np.array([0.09, 0.20, 0.09], dtype=np.float32),
    )
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick up the wine glass", num_steps=60,
    )
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, bbox, sprite=None)
    return sim, traj, bbox


# --------------------------------------------------------------- rasterizer

def test_backface_culling_orientation():
    """A culled box must show its NEAR face, not its far one.

    This pins down raster.FRONT_FACE_SIGN. Getting it backwards renders every
    closed solid inside-out.
    """
    box = P.box((0, 0, 0), (1, 1, 1), (1, 1, 1), material=Material(cull_backfaces=True))
    assert _nearest_depth(box, (0, 0, 3)) == pytest.approx(2.5, abs=0.02)


@pytest.mark.parametrize("mesh,eye,expected", [
    (P.box((0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)), (1.2, 0, 0), 0.95),
    (P.cylinder((0, 0, 0), 0.3, 0.4, (1, 1, 1), segments=48), (1.2, 0, 0), 0.90),
    (P.cylinder((0, 0, 0), 0.3, 0.4, (1, 1, 1), segments=48, caps=False), (1.2, 0, 0), 0.90),
    (P.uv_sphere((0, 0, 0), 0.3, (1, 1, 1), segments=40, rings=24), (1.2, 0, 0), 0.90),
    (P.tube(np.array([[0, -0.3, 0], [0, 0, 0], [0, 0.3, 0]], np.float32),
            np.array([0.12, 0.12, 0.12], np.float32), (1, 1, 1), segments=32),
     (1.2, 0, 0), 1.08),
])
def test_closed_primitives_face_outward(mesh, eye, expected):
    """Every closed primitive shows its outward surface under culling."""
    assert _nearest_depth(mesh, eye) == pytest.approx(expected, abs=0.03)


def test_cylinder_top_cap_faces_up():
    """Regression: inverted cap winding put the pedestal's underside on top of
    the bench, coplanar with it, producing z-fighting stripes."""
    cyl = P.cylinder((0, 0, 0), 0.3, 0.4, (1, 1, 1), segments=48)
    assert _nearest_depth(cyl, (0, 1.2, 0.0001), up=(0, 0, -1)) == pytest.approx(1.0, abs=0.02)


def test_perspective_correct_depth():
    """A quad slanted in depth must interpolate 1/z, not z."""
    quad = P.quad((-1, -1, 0), (1, -1, -2), (1, 1, -2), (-1, 1, 0), color=(1, 1, 1))
    cam = Camera(position=(0, 0, 4), target=(0, 0, -1), up=(0, 1, 0),
                 fov_y_deg=60, width=64, height=64)
    rast = Rasterizer(64, 64)
    rast.draw(quad, cam)
    row = rast.depth[32]
    finite = np.flatnonzero(np.isfinite(row))
    assert len(finite) > 20
    depths = row[finite]
    # Depth increases monotonically toward the receding edge...
    assert np.all(np.diff(depths) > -1e-4)
    # ...and non-linearly: a linear-in-screen-space interpolation (the classic
    # affine-texturing bug) would make the second differences vanish.
    assert np.abs(np.diff(depths, 2)).max() > 1e-4


def test_near_plane_clipping_does_not_explode():
    """Geometry straddling the near plane clips instead of wrapping around."""
    quad = P.quad((-2, -2, 1), (2, -2, 1), (2, 2, -1), (-2, 2, -1), color=(1, 1, 1))
    cam = Camera(position=(0, 0, 0.5), target=(0, 0, -1), up=(0, 1, 0),
                 fov_y_deg=60, width=64, height=64, near=0.05)
    rast = Rasterizer(64, 64)
    rast.draw(quad, cam)
    finite = rast.depth[np.isfinite(rast.depth)]
    assert len(finite) > 0
    assert finite.min() >= cam.near - 1e-4


def test_orient_faces_outward_is_idempotent():
    sphere = P.uv_sphere((0, 0, 0), 0.3, (1, 1, 1))
    once = orient_faces_outward(sphere)
    twice = orient_faces_outward(once)
    assert np.array_equal(once.faces, twice.faces)


# ---------------------------------------------------------------- hand mesh

def test_hand_mesh_tracks_keypoints():
    """The mesh must enclose the joints it was built from, at a sane size."""
    kpts = MockTrajectoryDiffusion()._generate_hand_keypoints_3d(
        np.zeros(3, np.float32), np.zeros(3, np.float32), 0.3)
    mesh = HM.build_hand_mesh(kpts)
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert np.all(kpts.min(axis=0) >= lo - 1e-3)
    assert np.all(kpts.max(axis=0) <= hi + 1e-3)
    # Skin adds volume, but not more than a couple of centimetres per side.
    assert np.all((kpts.min(axis=0) - lo) < 0.03)
    assert 200 < mesh.num_faces < 700


def test_hand_mesh_parts_individually_oriented():
    """Regression: orienting the MERGED hand tested fingertip faces against the
    palm centroid and flipped them, punching holes in the fingers."""
    kpts = MockTrajectoryDiffusion()._generate_hand_keypoints_3d(
        np.zeros(3, np.float32), np.zeros(3, np.float32), 0.3)
    lab = (kpts - kpts[0]) @ np.diag([1.0, -1.0, -1.0]).astype(np.float32).T
    mesh = HM.build_hand_mesh(lab)
    cam = Camera(position=(0, 0.05, 0.4), target=(0, 0.05, 0), up=(0, 1, 0),
                 fov_y_deg=45, width=160, height=160)
    rast = Rasterizer(160, 160)
    rast.draw(mesh, cam)
    assert (rast.gbuffer[:, :, 12] > 0.5).sum() > 800


# -------------------------------------------------------------- object mesh

def test_object_mesh_takes_aspect_from_silhouette():
    """Scale comes from the caller, SHAPE from the crop. Regression: trusting
    the detector's 3-D extent for aspect flattened a tall object into a lozenge.
    """
    sprite = np.full((160, 60, 3), 30, dtype=np.uint8)
    sprite[16:144, 12:48] = (40, 60, 200)          # a tall, narrow object
    mesh = OM.build_object_mesh(sprite, longest_dim_m=0.20)
    assert mesh is not None
    sx, sy, sz = OM.mesh_extent(mesh)
    assert sy == pytest.approx(0.20, abs=0.02)     # tall axis takes the scale
    assert sx < sy * 0.65                          # and stays narrow
    assert 0.0 < sz < min(sx, sy)


def test_object_mesh_falls_back_without_a_crop():
    mesh = OM.fallback_object_mesh(0.12)
    assert mesh.num_faces == 12
    assert OM.mesh_extent(mesh)[0] == pytest.approx(0.12, abs=1e-4)


# ------------------------------------------------------------------ lab sim

def test_lab_transform_flips_perception_axes(staged_sim):
    """Perception is +Y down / +Z away; the lab is +Y up / +Z toward the viewer."""
    sim, _, bbox = staged_sim
    xf = sim.transform
    centre = xf(bbox.center)[0]
    assert centre == pytest.approx(xf.anchor_lab, abs=1e-4)

    # One centimetre "up" in perception (-Y) must be up in the lab (+Y).
    up_cam = bbox.center + np.array([0.0, -0.01, 0.0], dtype=np.float32)
    assert xf(up_cam)[0][1] > centre[1]
    # One centimetre further from the camera (+Z) must go deeper (-Z) in the lab.
    far_cam = bbox.center + np.array([0.0, 0.0, 0.01], dtype=np.float32)
    assert xf(far_cam)[0][2] < centre[2]


def test_object_rests_on_the_stage(staged_sim):
    """At rest the object's base sits exactly on the pedestal - not sunk, not
    floating. Regression: orienting the mesh AFTER measuring its height left it
    hovering."""
    sim, _, _ = staged_sim
    seated = sim.object_mesh.transformed(translation=sim._object_path_lab[0])
    assert float(seated.vertices[:, 1].min()) == pytest.approx(LS.PEDESTAL_TOP_Y, abs=2e-3)


def test_object_is_lifted_by_the_plan(staged_sim):
    sim, traj, _ = staged_sim
    lift = sim.telemetry(len(traj.waypoints) - 1)["lift_cm"]
    assert lift > 4.0


def _assert_in_viewport(sim, points, step):
    screen, depth = sim.camera.project(points)
    assert np.all(depth > sim.camera.near), f"behind the near plane at step {step}"
    assert screen[:, 0].min() >= 0 and screen[:, 0].max() <= sim.width - 1, f"clipped in x at step {step}"
    assert screen[:, 1].min() >= 0 and screen[:, 1].max() <= sim.height - 1, f"clipped in y at step {step}"


def _object_corners(sim, step):
    half = sim.object_size.astype(np.float32) * 0.5
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                       dtype=np.float32) * half
    return sim._object_path_lab[step] + corners


def test_object_never_leaves_the_viewport(staged_sim):
    """The manipuland must stay in frame for the whole plan, including its lift.

    The hand is deliberately NOT asserted here: the camera is locked, so a plan
    whose approach starts far from the bench enters from off-screen by design.
    The object is staged on the pedestal and has nowhere else to be.
    """
    sim, traj, _ = staged_sim
    for step in range(len(traj.waypoints)):
        _assert_in_viewport(sim, _object_corners(sim, step), step)


def test_the_hand_is_in_frame_for_the_grasp_and_the_lift(staged_sim):
    """Whatever the approach does, the business end must be visible."""
    sim, traj, _ = staged_sim
    grasp = sim._contact_step()
    for step in range(grasp, len(traj.waypoints)):
        _assert_in_viewport(sim, sim._hand_paths_lab[step], step)


def test_hand_is_large_enough_to_read(staged_sim):
    """The locked camera must still leave the hand a meaningful fraction of the
    viewport at the grasp, or the reenactment is unreadable however correct."""
    sim, _, _ = staged_sim
    assert sim.hand_screen_height(sim._contact_step()) > 0.18 * sim.height


def test_camera_is_locked_across_different_objects_and_plans():
    """The whole point of the locked shot: two different manipulands with two
    different plans must be filmed from exactly the same pose."""
    poses = []
    for size in [(0.05, 0.05, 0.05), (0.39, 0.39, 0.20)]:
        bbox = BoundingBox3D(label="x", center=np.array([0.03, 0.02, 0.55], np.float32),
                             size=np.array(size, dtype=np.float32))
        traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
            start_hand_pose=None, target_object=bbox,
            affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
            intent="pick", num_steps=60)
        sim = LabSimulator(width=192, height=144)
        assert sim.prepare(traj, bbox, sprite=None)
        poses.append((tuple(np.round(sim.camera.position, 5)),
                      tuple(np.round(sim.camera.target, 5)),
                      round(sim.camera.fov_y_deg, 5)))
    assert poses[0] == poses[1], f"camera moved between plans: {poses}"


def test_the_object_visibly_rises_when_it_is_picked_up(staged_sim):
    """A lift the viewer cannot see is a lift that did not happen.

    Regression from a live run reported as 'it sat there and never moved': the
    object did rise, but only ~37 px, only over the last third of the rollout,
    and with no grasp to cue it.
    """
    sim, traj, _ = staged_sim
    top = sim.camera.project(sim._object_path_lab[0][None, :])[0][0, 1]
    bottom = sim.camera.project(sim._object_path_lab[-1][None, :])[0][0, 1]
    rise_px = float(top - bottom)
    assert rise_px > 0.15 * sim.height, f"object rose only {rise_px:.0f}px"


def test_render_produces_a_lit_image(staged_sim):
    sim, _, _ = staged_sim
    img = sim.render(sim._contact_step())
    assert img is not None and img.shape == (sim.height, sim.width, 3)
    assert img.dtype == np.uint8
    assert img.std() > 12.0            # not a flat fill
    assert 20 < img.mean() < 235       # neither crushed nor blown out


def test_render_is_throttled_between_identical_steps(staged_sim):
    """Back-to-back calls for the same step reuse the frame instead of spending
    the client's budget re-rendering it."""
    sim, _, _ = staged_sim
    step = sim._contact_step()
    sim._last_render_t = 0.0
    first = sim.render(step, push_in=0.5)
    sim.last_render_ms = -1.0
    second = sim.render(step, push_in=0.5)
    assert second is first
    assert sim.last_render_ms == -1.0  # untouched: no re-render happened


def test_prepare_rejects_an_empty_plan():
    sim = LabSimulator(width=96, height=72)
    assert sim.prepare(None, None, None) is False
    assert sim.is_ready is False
    assert sim.render(0) is None


def test_view_direction_points_up_and_toward_the_viewer():
    d = _view_direction()
    assert d[1] > 0 and d[2] > 0
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-5)


def test_shading_respects_shadow_attenuation():
    """A shadowed sample must be darker than the same sample unshadowed, but not
    black - ambient still reaches it."""
    rast = Rasterizer(8, 8)
    rast.draw(P.quad((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1), color=(0.6, 0.6, 0.6)),
              Camera(position=(0, 1.5, 0.001), target=(0, 0, 0), up=(0, 0, -1),
                     fov_y_deg=60, width=8, height=8))
    cam = Camera(position=(0, 1.5, 0.001), target=(0, 0, 0), up=(0, 0, -1),
                 fov_y_deg=60, width=8, height=8)
    lights = [SH.Light(direction=(0, -1, 0), color=(1, 1, 1), intensity=1.0)]
    env = SH.Environment(sky_color=(0.2, 0.2, 0.2), ground_color=(0.05, 0.05, 0.05),
                         fog_color=(0, 0, 0), fog_start=10, fog_end=20)
    rows = rast.gbuffer.reshape(-1, 13)
    depth = rast.depth.reshape(-1)
    lit = SH.shade_rows(rows, depth, cam, lights, env)
    shadowed = SH.shade_rows(rows, depth, cam, lights, env,
                             shadow_rows=np.zeros(len(rows), dtype=np.float32))
    covered = rows[:, 12] > 0.5
    assert covered.any()
    assert shadowed[covered].mean() < lit[covered].mean()
    assert shadowed[covered].mean() > 0.0


# ------------------------------------------------- object scale sanity (prod)

def test_object_scale_is_bounded_by_the_grasping_hand():
    """Regression from the RunPod test: the detector reported a wine glass at
    over 34 cm, staging a beach-ball on the bench. The detector's extent is
    back-projected from non-metric depth, so it is bounded by the one metric
    reference in the scene - the hand doing the grasping."""
    from src.simulation.lab_sim import _object_longest_dimension, _REFERENCE_PALM_M

    huge = BoundingBox3D(label="wine glass", center=np.zeros(3, np.float32),
                         size=np.array([0.27, 0.34, 0.15], dtype=np.float32))
    staged = _object_longest_dimension(huge, _REFERENCE_PALM_M)
    assert staged <= 2.2 * _REFERENCE_PALM_M + 1e-6
    assert staged < 0.34

    tiny = BoundingBox3D(label="pen", center=np.zeros(3, np.float32),
                         size=np.array([0.004, 0.004, 0.004], dtype=np.float32))
    assert _object_longest_dimension(tiny, _REFERENCE_PALM_M) >= 0.45 * _REFERENCE_PALM_M - 1e-6

    ok = BoundingBox3D(label="mug", center=np.zeros(3, np.float32),
                       size=np.array([0.10, 0.12, 0.10], dtype=np.float32))
    assert _object_longest_dimension(ok, _REFERENCE_PALM_M) == pytest.approx(0.12, abs=1e-6)

    # No detector box at all still yields a plausible manipuland.
    assert 0.05 <= _object_longest_dimension(None, _REFERENCE_PALM_M) <= 0.20


def test_oversized_detection_still_frames_readably():
    """With the clamp in place, an absurd detector box must not blow out the shot."""
    bbox = BoundingBox3D(label="wine glass", center=np.array([0.02, 0.05, 0.46], np.float32),
                         size=np.array([0.27, 0.34, 0.15], dtype=np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick up the wine glass", num_steps=60)
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, bbox, sprite=None)
    # The object must fit on the staging pedestal it is sitting on.
    assert float(sim.object_size.max()) < 2 * 0.135
    # A max-size manipuland is the worst case for framing: the camera must pull
    # back to keep it in shot, but the hand still has to read.
    assert sim.hand_screen_height(sim._contact_step()) > 0.20 * sim.height
    half = sim.object_size.astype(np.float32) * 0.5
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                       dtype=np.float32) * half
    for step in (0, sim._contact_step(), len(traj.waypoints) - 1):
        _assert_in_viewport(sim, sim._object_path_lab[step] + corners, step)


# ------------------------------------------------------- the grasp must grasp

def test_the_thumb_opposes_the_fingers():
    """Without opposition the hand can rest on an object but never hold one.

    Reported live as 'the ghost hand must pick up the object as well'. In the
    canonical pose every digit extends along the same axis, so the thumb was
    just a shorter finger in the fingers' own plane.
    """
    gen = MockTrajectoryDiffusion()
    closed = gen._generate_hand_keypoints_3d(
        np.zeros(3, np.float32), gen._rot_grasp, gen._flex_closed)
    opened = gen._generate_hand_keypoints_3d(
        np.zeros(3, np.float32), gen._rot_grasp, 0.0)

    def gap(k):
        return float(np.linalg.norm(k[4] - k[[8, 12]].mean(axis=0)))

    assert gap(closed) < gap(opened), "closing the hand must bring thumb and fingers together"
    assert gap(closed) < 0.07, f"thumb never reaches the fingers (gap {gap(closed):.3f} m)"


def test_closed_hand_does_not_splay_wider_than_a_palm():
    """At the grasp the fingertips must span roughly a hand, not a dinner plate.

    The fan that stops the fingers collapsing into one slab peaks exactly at
    contact if it is not damped: tips ended up 13.8 cm apart around a 4.4 cm
    object, closing on empty air either side of it.
    """
    gen = MockTrajectoryDiffusion()
    closed = gen._generate_hand_keypoints_3d(
        np.zeros(3, np.float32), gen._rot_grasp, gen._flex_closed)
    palm = float(np.linalg.norm(closed[9] - closed[0]))
    spread = float(np.linalg.norm(closed[[4, 8, 12, 16, 20]].ptp(axis=0)))
    assert spread < 1.4 * palm, \
        f"fingertips span {spread*100:.1f} cm on a {palm*100:.1f} cm palm"


@pytest.mark.parametrize("detector_size", [
    (0.11, 0.12, 0.11),      # a plausible cup
    (0.39, 0.39, 0.20),      # what the detector actually reported for one
    (0.15, 0.055, 0.03),     # a remote
])
def test_grasp_lands_on_the_object_as_staged(detector_size):
    """The planner sizes its approach from the detector's raw extent, but the lab
    stages a scale-corrected object. Un-reconciled, the hand closes in the air
    above a smaller object."""
    bbox = BoundingBox3D(label="x", center=np.array([0.03, 0.02, 0.55], np.float32),
                         size=np.array(detector_size, dtype=np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick", num_steps=60)
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, bbox, sprite=None)

    grasp = sim._contact_step()
    tips = sim._hand_paths_lab[grasp][[4, 8, 12, 16, 20]]
    pinch = 0.5 * (tips[0] + tips[[1, 2]].mean(axis=0))
    reach = float(np.linalg.norm(pinch - sim._object_path_lab[grasp]))
    assert reach < 0.6 * float(sim.object_size.max()), \
        f"grasp closed {reach*100:.1f} cm from the object it is meant to be holding"


def test_the_object_travels_with_the_hand_once_grasped():
    """After contact the object and hand must rise together, not drift apart."""
    bbox = BoundingBox3D(label="cup", center=np.array([0.03, 0.02, 0.55], np.float32),
                         size=np.array([0.11, 0.12, 0.11], dtype=np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick", num_steps=60)
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, bbox, sprite=None)

    grasp = sim._contact_step()
    last = len(traj.waypoints) - 1
    hand_rise = float(sim._hand_paths_lab[last][0, 1] - sim._hand_paths_lab[grasp][0, 1])
    obj_rise = float(sim._object_path_lab[last][1] - sim._object_path_lab[grasp][1])
    assert obj_rise > 0.05, f"object barely lifted ({obj_rise*100:.1f} cm)"
    assert abs(hand_rise - obj_rise) < 0.02, \
        f"hand rose {hand_rise*100:.1f} cm but object rose {obj_rise*100:.1f} cm - they separated"
