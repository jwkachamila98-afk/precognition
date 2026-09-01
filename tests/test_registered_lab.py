"""The registered lab renders into the real camera (tests/test_registered_lab.py).

Registration is a claim about pixels: the object drawn in the reenactment must
land where the detector says it is, so the reenactment can be composited over
the live frame without anything sliding. So it is checked in pixels, against
the same projection the live overlay draws its box with.

The studio mode is deliberately left alone and is covered by test_lab_renderer.
"""

import math

import numpy as np
import pytest

from src.perception.hand_tracker import HandPose, HandSide
from src.perception.scene_parser import BoundingBox3D
from src.simulation.lab_sim import LabSimulator
from src.simulation.render.camera import Camera

W, H = 320, 240
AXIS_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _bbox(label="cup"):
    return BoundingBox3D(label=label,
                         center=np.array([0.07, -0.04, 0.62], dtype=np.float32),
                         size=np.array([0.10, 0.11, 0.10], dtype=np.float32))


def _demonstration(bbox, n=40):
    """A reach that ends at the object, in camera-frame metres."""
    poses = []
    start = np.array([-0.16, 0.12, 0.55], dtype=np.float32)
    rng = np.random.default_rng(0)
    palm = rng.normal(0, 0.03, (21, 3)).astype(np.float32)
    for i in range(n):
        t = i / (n - 1)
        wrist = start + (bbox.center - start) * (1 - (1 - t) ** 2)
        k3 = (palm + wrist).astype(np.float32)
        fx = 0.8 * W
        z = np.maximum(k3[:, 2], 1e-6)
        k2 = np.stack([fx * k3[:, 0] / z + W / 2,
                       fx * k3[:, 1] / z + H / 2], 1).astype(np.float32)
        poses.append(HandPose(hand_id=0, side=HandSide.RIGHT, keypoints_3d=k3,
                              keypoints_2d=k2, confidence=0.9, timestamp=i / 30.0))
    return poses


def test_the_registered_camera_reproduces_the_projection_convention():
    """The whole feature rests on this: a lab camera at the origin with a field
    of view derived from fx = 0.8*w must agree with BoundingBox3D.project_to_2d
    to the pixel. If it does not, everything downstream is fitted rather than
    registered."""
    sim = LabSimulator(width=W, height=H, registered=True)
    cam = sim._registered_camera()

    bbox = _bbox()
    expected = bbox.project_to_2d(image_shape=(H, W))
    got, _ = cam.project(bbox.corners_3d @ AXIS_FLIP.T)

    assert np.abs(expected - got).max() < 1e-3, (
        "the registered camera and the projection convention disagree")


def test_the_rendered_object_lands_where_the_detector_put_it():
    """Before contact the object has not moved, so its rendered pixels must sit
    on the detected box. Checked by centroid and containment rather than by
    exact extent: the mesh is the object's shape, not its bounding box, so a
    turned cup is legitimately narrower than the box that contains it."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)

    sim._rast.clear()
    sim._rast.draw(sim._object_mesh_for(0.0), sim.camera)
    ys, xs = np.nonzero(sim._rast.gbuffer[:, :, 12] > 0.5)
    assert len(xs) > 0, "the object rendered no pixels at all"

    corners = bbox.project_to_2d(image_shape=(H, W))
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()

    # Centred on the detection.
    cx_err = abs((xs.min() + xs.max()) / 2 - (x0 + x1) / 2)
    cy_err = abs((ys.min() + ys.max()) / 2 - (y0 + y1) / 2)
    span = max(x1 - x0, y1 - y0)
    assert cx_err < 0.12 * span and cy_err < 0.12 * span, (
        f"rendered object is off-centre by ({cx_err:.1f}, {cy_err:.1f}) px "
        f"against a {span:.1f} px detection")

    # And contained by it, allowing a pixel of rasterisation slop.
    assert xs.min() >= x0 - 2 and xs.max() <= x1 + 2
    assert ys.min() >= y0 - 2 and ys.max() <= y1 + 2


def test_registered_rendering_composites_onto_the_supplied_frame():
    """The background is the live frame, not a baked studio. Pixels far from the
    actors must therefore survive untouched."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)

    background = np.full((H, W, 3), (37, 43, 61), np.uint8)
    out = sim.render(0.0, elapsed=0.0, background=background)
    assert out is not None and out.shape == background.shape

    changed = np.any(out != background, axis=2)
    assert changed.any(), "nothing was drawn over the frame"
    assert not changed.all(), "the whole frame was overpainted - the live image is gone"
    # The far corner is nowhere near the object or the reach.
    assert not changed[H - 1, 0], "a corner of the live frame was overpainted"


def test_registered_mode_refuses_to_invent_a_background():
    """Without a frame there is nothing to register against, and returning a
    studio render instead would silently drop the mode."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)
    assert sim.render(0.0, elapsed=0.0) is None


def test_the_studio_mode_still_stages_the_old_way():
    """The studio is switched off, not deleted: it still anchors the object to
    the bench and normalises the hand, which is what makes every reenactment
    frame identically."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=False)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)
    assert sim.transform.scale != 1.0 or not np.array_equal(
        sim.transform.origin_cam, np.zeros(3, dtype=np.float32)), (
        "the studio path is staging with the registered identity transform")
    assert sim.render(0.0, elapsed=0.0) is not None, "studio render needs no frame"


def _timed_demonstration(bbox, n=120, fps=8.0):
    """A reach whose wall-clock duration is set by its timestamps."""
    poses = _demonstration(bbox, n)
    return [HandPose(hand_id=p.hand_id, side=p.side, keypoints_3d=p.keypoints_3d,
                     keypoints_2d=p.keypoints_2d, confidence=p.confidence,
                     timestamp=i / fps)
            for i, p in enumerate(poses)]


@pytest.mark.parametrize("fps,label", [(8.0, "longer than the window"),
                                       (40.0, "shorter than the window")])
def test_the_whole_recording_plays(fps, label):
    """The replay must reach the last recorded frame by the end of the demo.

    Playback mapped wall-clock straight onto recorded time, which is real speed
    - correct until the reach outlasts the demo window, at which point it simply
    stopped wherever the phase ran out. A 15 s reach in a 12 s window lost its
    last quarter, grasp included, while still being presented as a replay of the
    attempt. Long recordings are now compressed to fit; short ones still play at
    real speed.
    """
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    sim.demo_duration_sec = 12.0
    assert sim.prepare_from_demonstration(_timed_demonstration(bbox, fps=fps), bbox, None)

    last = sim._num_steps - 1
    assert sim.step_for_progress(1.0) >= last - 0.5, (
        f"a recording {label} did not reach its final frame: "
        f"stopped at {sim.step_for_progress(1.0):.1f} of {last}")
    # And it is monotonic, so nothing plays backwards on the way there.
    steps = [sim.step_for_progress(p) for p in np.linspace(0.0, 1.0, 40)]
    assert all(b >= a - 1e-6 for a, b in zip(steps, steps[1:]))


def test_a_short_recording_still_plays_at_real_speed():
    """Compression is for recordings that do not fit. A three-second reach must
    not be stretched across a twelve-second window."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    sim.demo_duration_sec = 12.0
    assert sim.prepare_from_demonstration(
        _timed_demonstration(bbox, n=90, fps=30.0), bbox, None)   # 3.0 s
    # A third of the way through a 12 s phase is 4 s of wall clock, by which
    # point a 3 s recording is finished.
    assert sim.step_for_progress(1.0 / 3.0) >= sim._num_steps - 1.5


def test_the_stylised_pad_sits_under_the_object_and_is_scaled_to_it():
    """A floor is the wrong metaphor. A horizontal plane large enough to read as
    ground runs to the horizon from a camera at the origin looking level: at
    1.3 m it covered the upper half of the frame and left the actors a sliver.
    The pad is scaled to the object so it stages without dominating, and sits at
    the object's underside, which is the one height we actually know - the object
    is resting on it.
    """
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)
    assert sim.set_stylised_room()

    mesh = sim.scene_mesh
    assert mesh is not None
    ys = mesh.vertices[:, 1]
    base = sim._object_path_lab[0][1] - sim.object_size[1] * 0.5
    assert np.allclose(ys, base, atol=1e-4), "the pad is not at the object's base"

    span = float(np.linalg.norm(
        mesh.vertices[:, [0, 2]].max(axis=0) - mesh.vertices[:, [0, 2]].min(axis=0)))
    footprint = float(np.max(sim.object_size[[0, 2]]))
    assert span < 12.0 * footprint, f"pad spans {span:.2f} m for a {footprint:.2f} m object"


def test_the_stylised_set_replaces_the_photograph_rather_than_covering_it():
    """Its ground fades to black, which only vanishes against black. Composited
    over a lighter camera frame the same fade became a hard dark bar cutting
    across the actors, which is what it looked like when first tried."""
    bbox = _bbox()
    sim = LabSimulator(width=W, height=H, registered=True)
    assert sim.prepare_from_demonstration(_demonstration(bbox), bbox, None)
    assert sim.set_stylised_room()

    background = np.full((H, W, 3), (200, 200, 200), np.uint8)   # a bright room
    out = sim.render(0.0, elapsed=0.0, background=background)
    assert out is not None
    # None of that brightness may survive: the set is not a layer on the video.
    assert out.mean() < 90, "the photograph is showing through the stylised set"
