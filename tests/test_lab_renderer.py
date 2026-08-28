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
    """Enough of the hand must SURVIVE OCCLUSION at the grasp to read as a hand.

    Deliberately not screen height, which this used to assert. Height is a poor
    proxy: an approach angled toward the viewer is shorter in screen-Y while
    showing more of itself, and the angle that best aligns the palm to the
    camera puts the hand behind the object, where measured visibility collapsed
    to a quarter of its best value while height barely moved.
    """
    sim, _, _ = staged_sim
    visible = sim.visible_hand_fraction(sim._contact_step())
    assert visible > 0.012, f"only {visible:.2%} of the viewport shows hand"


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


def test_identical_frames_are_never_re_rendered(staged_sim):
    """The same step and zoom must reuse the cached frame unconditionally.

    Regression: this was additionally gated on a rate-limit interval measured
    from when the previous render STARTED, so as soon as a render took longer
    than the interval - which happens the moment the machine is under load - the
    cache stopped engaging and identical frames were redrawn from scratch.
    """
    sim, _, _ = staged_sim
    step = sim._contact_step()
    sim._last_render_t = 0.0
    first = sim.render(step, push_in=0.5)
    sim.last_render_ms = -1.0
    second = sim.render(step, push_in=0.5)
    assert second is first
    assert sim.last_render_ms == -1.0  # untouched: no re-render happened

    # Even with the rate limit long expired, identical content is still reused.
    sim._last_render_t = 0.0
    assert sim.render(step, push_in=0.5) is first
    assert sim.last_render_ms == -1.0


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
    from src.simulation.lab_sim import (_object_longest_dimension, _REFERENCE_PALM_M,
                                        _MAX_OBJECT_PALMS)

    # Unrecognised label, so this exercises the clamp rather than the class prior.
    huge = BoundingBox3D(label="aardvark", center=np.zeros(3, np.float32),
                         size=np.array([0.27, 0.34, 0.15], dtype=np.float32))
    staged = _object_longest_dimension(huge, _REFERENCE_PALM_M)
    assert staged <= _MAX_OBJECT_PALMS * _REFERENCE_PALM_M + 1e-6
    assert staged < 0.34

    tiny = BoundingBox3D(label="pen", center=np.zeros(3, np.float32),
                         size=np.array([0.004, 0.004, 0.004], dtype=np.float32))
    assert _object_longest_dimension(tiny, _REFERENCE_PALM_M) >= 0.45 * _REFERENCE_PALM_M - 1e-6

    # An unrecognised label has no prior, so a plausible measured extent passes
    # through untouched.
    ok = BoundingBox3D(label="aardvark", center=np.zeros(3, np.float32),
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
    # A max-size manipuland is the worst case: it is the most likely to occlude
    # the hand that is grasping it.
    assert sim.visible_hand_fraction(sim._contact_step()) > 0.010
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



# ------------------------------------------------------ class size priors

def test_class_prior_beats_the_detectors_non_metric_extent():
    """A recognised class is sized from its prior, not from back-projected depth.

    Both numbers below were measured live: 56 cm for a coffee cup on synthetic
    depth here, 74 cm for a wine glass on real MiDaS depth on a GPU pod. The
    hand-based clamp caught them, but only by pinning to its own ceiling, which
    still staged a mug at double life size.
    """
    from src.simulation.lab_sim import _object_longest_dimension, _REFERENCE_PALM_M

    cup = BoundingBox3D(label="cup", center=np.zeros(3, np.float32),
                        size=np.array([0.56, 0.56, 0.30], dtype=np.float32))
    staged = _object_longest_dimension(cup, _REFERENCE_PALM_M)
    assert staged == pytest.approx(0.09, abs=1e-6), "a cup is a cup, not 56 cm"

    glass = BoundingBox3D(label="wine glass", center=np.zeros(3, np.float32),
                          size=np.array([0.74, 0.74, 0.40], dtype=np.float32))
    assert _object_longest_dimension(glass, _REFERENCE_PALM_M) == pytest.approx(0.20, abs=1e-6)


def test_prior_lookup_prefers_the_most_specific_label():
    """'wine glass' must not be shadowed by a shorter substring match."""
    from src.simulation.lab_sim import _size_prior_for

    assert _size_prior_for("wine glass") == pytest.approx(0.20)
    assert _size_prior_for("cell phone") == pytest.approx(0.15)
    assert _size_prior_for("coffee cup") == pytest.approx(0.09)
    assert _size_prior_for("remote_control") == pytest.approx(0.16)
    assert _size_prior_for("aardvark") is None
    assert _size_prior_for(None) is None
    assert _size_prior_for("") is None


def test_unknown_classes_still_fall_back_to_the_detector():
    """The prior is an override for things we know, not a replacement for
    detection - an unrecognised label still uses the measured extent."""
    from src.simulation.lab_sim import _object_longest_dimension, _REFERENCE_PALM_M

    odd = BoundingBox3D(label="aardvark", center=np.zeros(3, np.float32),
                        size=np.array([0.12, 0.06, 0.06], dtype=np.float32))
    assert _object_longest_dimension(odd, _REFERENCE_PALM_M) == pytest.approx(0.12, abs=1e-6)


def test_priors_are_all_graspable():
    """Every prior must survive the hand clamp unchanged, or it is not a size a
    hand could be picking up and does not belong in the table."""
    from src.simulation.lab_sim import (_CLASS_SIZE_PRIORS_M, _REFERENCE_PALM_M,
                                        _MIN_OBJECT_PALMS, _MAX_OBJECT_PALMS)

    lo = _MIN_OBJECT_PALMS * _REFERENCE_PALM_M
    hi = _MAX_OBJECT_PALMS * _REFERENCE_PALM_M
    for label, size in _CLASS_SIZE_PRIORS_M.items():
        assert lo <= size <= hi, f"prior for '{label}' ({size*100:.0f} cm) is outside [{lo*100:.0f}, {hi*100:.0f}] cm"


# ------------------------------------------------------------- smooth motion

def test_plan_is_sampled_continuously_not_snapped(staged_sim):
    """The trajectory position must be fractional.

    Rounding it to a whole waypoint quantised the reenactment to 60 discrete
    poses, which reads as a stutter no matter how smoothly progress advances.
    """
    sim, _, _ = staged_sim
    positions = [sim.step_for_progress(p) for p in np.linspace(0.0, 0.82, 40)]
    assert any(abs(p - round(p)) > 1e-3 for p in positions), "positions are snapped to whole waypoints"
    assert all(b >= a - 1e-6 for a, b in zip(positions, positions[1:])), "must advance monotonically"


def test_interpolated_pose_lies_between_its_neighbours(staged_sim):
    """A fractional index must blend the two waypoints it sits between."""
    sim, _, _ = staged_sim
    a = sim._lerp_along(sim._hand_paths_lab, 10.0)
    b = sim._lerp_along(sim._hand_paths_lab, 11.0)
    mid = sim._lerp_along(sim._hand_paths_lab, 10.5)
    assert np.allclose(a, sim._hand_paths_lab[10])
    assert np.allclose(b, sim._hand_paths_lab[11])
    assert np.allclose(mid, 0.5 * (a + b), atol=1e-5)
    assert not np.allclose(mid, a), "midpoint must differ from either neighbour"


def test_motion_between_frames_stays_small_at_client_frame_rates(staged_sim):
    """No frame may lurch. Driving the animation from the server's phase
    progress - which lands about once a second - and snapping to whole waypoints
    produced 60 px jumps between otherwise-frozen frames.
    """
    sim, _, _ = staged_sim
    fps, demo_seconds = 8.0, 6.0
    prev, jumps = None, []
    for i in range(int(fps * demo_seconds)):
        pos = sim.step_for_progress(min((i / fps) / demo_seconds, 1.0))
        screen = sim.camera.project(sim._lerp_along(sim._hand_paths_lab, pos))[0]
        if prev is not None:
            jumps.append(float(np.abs(screen - prev).max()))
        prev = screen
    assert max(jumps) < 0.12 * sim.height, f"largest single-frame jump {max(jumps):.0f}px"


def test_telemetry_reads_a_fractional_position(staged_sim):
    """The HUD must accept the same index the renderer uses, and its continuous
    fields must not tick in whole-waypoint steps."""
    sim, traj, _ = staged_sim
    mid = sim._contact_step() + 0.5
    tel = sim.telemetry(mid)
    assert 1 <= tel["step"] <= len(traj.waypoints)
    lifts = [sim.telemetry(s)["lift_cm"] for s in np.linspace(40.0, 59.0, 12)]
    assert any(abs(v - round(v, 1)) > 1e-9 for v in lifts) or len(set(lifts)) > 6


# ------------------------------------------------- class-parametric geometry

def test_known_classes_get_real_geometry_not_an_inflated_silhouette():
    """A bottle should be a bottle. Silhouette inflation makes no assumption
    about the object, which is right for the unrecognised and wrong for
    everything else - a single view has no depth in it, so a bottle came out a
    lumpy slab. The class is better evidence than that measurement.
    """
    from src.simulation.render.object_library import build_class_mesh

    bottle = build_class_mesh("water bottle", 0.25)
    assert bottle is not None and bottle.num_faces > 100
    sx, sy, sz = (bottle.vertices.max(axis=0) - bottle.vertices.min(axis=0))
    assert sy == pytest.approx(0.25, abs=0.01), "height must match the staged size"
    assert sx == pytest.approx(sz, abs=0.01), "a turned object is round in plan"
    assert sx < 0.5 * sy, "a bottle is taller than it is wide"

    # The neck must actually be narrower than the body.
    v = bottle.vertices
    body = v[v[:, 1] < v[:, 1].min() + 0.10]
    neck = v[v[:, 1] > v[:, 1].max() - 0.03]
    assert np.hypot(neck[:, 0], neck[:, 2]).max() < np.hypot(body[:, 0], body[:, 2]).max()


def test_unknown_classes_fall_through_to_the_silhouette():
    from src.simulation.render.object_library import build_class_mesh
    assert build_class_mesh("aardvark", 0.2) is None
    assert build_class_mesh(None, 0.2) is None


def test_flat_classes_lie_down_on_the_bench():
    """A remote rests on its back; its long axis is horizontal, not vertical."""
    from src.simulation.render.object_library import build_class_mesh
    remote = build_class_mesh("remote control", 0.16)
    extent = remote.vertices.max(axis=0) - remote.vertices.min(axis=0)
    assert extent[1] < extent[2], "a remote should not be standing on end"


def test_colour_comes_from_the_real_crop():
    """Shape is canonical, so colour is what keeps the object recognisable."""
    from src.simulation.render.object_library import build_class_mesh, colour_ramp

    sprite = np.zeros((60, 20, 3), dtype=np.uint8)
    sprite[:20] = (40, 40, 200)          # red cap (BGR), image top
    sprite[20:] = (60, 180, 60)          # green body
    ramp = colour_ramp(sprite)
    assert ramp is not None
    assert ramp[-1][2] > ramp[-1][1], "mesh top should take the cap's red"
    assert ramp[0][1] > ramp[0][2], "mesh bottom should take the body's green"

    mesh = build_class_mesh("water bottle", 0.25, sprite)
    top = mesh.colors[mesh.vertices[:, 1] > mesh.vertices[:, 1].max() - 0.01]
    bottom = mesh.colors[mesh.vertices[:, 1] < mesh.vertices[:, 1].min() + 0.01]
    assert top[:, 2].mean() > bottom[:, 2].mean(), "colour ramp not applied bottom-to-top"


def test_class_mesh_is_preferred_over_the_silhouette():
    """A recognised class must not go through inflation even with a crop present."""
    bbox = BoundingBox3D(label="water bottle", center=np.array([0.03, 0.02, 0.55], np.float32),
                         size=np.array([0.09, 0.28, 0.09], dtype=np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick", num_steps=60)
    sprite = np.full((80, 30, 3), 120, dtype=np.uint8)
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, bbox, sprite)
    assert sim.object_is_reconstructed is False, "should have used the class mesh"
    assert sim.object_size[1] > sim.object_size[0], "bottle staged upright"


def test_geometry_follows_the_plans_target_not_the_frames_detection():
    """The staged object must match the plan being reenacted.

    The client stages on a later frame than the one the plan was generated from,
    so the detector may by then be reporting a different object. Taking the label
    from the frame staged one object's geometry against another object's
    trajectory - seen live as a bottle plan rendered with a stale, unrecognised
    box at 4x12x5 cm instead of a 7x25x7 cm bottle.
    """
    bbox = BoundingBox3D(label="water bottle", center=np.array([0.03, 0.02, 0.55], np.float32),
                         size=np.array([0.09, 0.28, 0.09], dtype=np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick", num_steps=60)

    # The frame the client happens to stage on now shows something else entirely.
    stale = BoundingBox3D(label="aardvark", center=np.array([0.03, 0.02, 0.55], np.float32),
                          size=np.array([0.12, 0.05, 0.05], dtype=np.float32))
    sim = LabSimulator(width=192, height=144)
    assert sim.prepare(traj, stale, sprite=None)

    assert sim.object_is_reconstructed is False, "should still use the bottle's class mesh"
    assert sim.object_size[1] == pytest.approx(0.25, abs=0.02), \
        f"staged {sim.object_size.round(3)} - took its size from the stale detection"
    assert sim.object_size[1] > 2.5 * sim.object_size[0], "a bottle, not a lozenge"


def test_the_hand_never_approaches_inverted():
    """Fingertips must stay below the wrist for a top-down grasp, at every step.

    Reported live as "the ghost hand is inverted". The approach interpolated the
    wrist orientation from an incompatible starting triple - the identity, or
    the tracker's [pitch, yaw, 0] palm direction, neither of which is the same
    parameterisation as the grasp pose where rx = 2.85 encodes "inverted". The
    hand entered upside down with its fingertips 14 cm ABOVE the wrist and
    cartwheeled 163 degrees on the way in.
    """
    from src.mocks.mock_hand_tracker import MockHandTracker
    tips = [4, 8, 12, 16, 20]

    for hand in (None, MockHandTracker().estimate(np.zeros((480, 640, 3), np.uint8))[0]):
        bbox = BoundingBox3D(label="water bottle", center=np.array([0.03, 0.02, 0.55], np.float32),
                             size=np.array([0.09, 0.28, 0.09], dtype=np.float32))
        traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
            start_hand_pose=hand, target_object=bbox,
            affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
            intent="pick", num_steps=60)
        sim = LabSimulator(width=192, height=144)
        assert sim.prepare(traj, bbox, sprite=None)

        for step in range(len(traj.waypoints)):
            pose = sim._hand_paths_lab[step]
            assert float(pose[tips][:, 1].mean()) < float(pose[0][1]), (
                f"step {step}: fingertips above the wrist "
                f"({'no hand' if hand is None else 'hand'} detected)")


def test_the_wrist_does_not_tumble_during_the_approach():
    """Total wrist rotation across the plan must stay modest. A hand orients
    itself as it reaches; 163 degrees of roll is a cartwheel."""
    gen = MockTrajectoryDiffusion()
    bbox = BoundingBox3D(label="water bottle", center=np.array([0.03, 0.02, 0.55], np.float32),
                         size=np.array([0.09, 0.28, 0.09], dtype=np.float32))
    traj = gen.generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick"),
        intent="pick", num_steps=60)
    rots = np.stack([wp.wrist_pose[3:6] for wp in traj.waypoints])
    travel = float(np.abs(rots[-1] - rots[0]).max())
    assert travel < np.radians(45.0), f"wrist rotated {np.degrees(travel):.0f} deg during the plan"
