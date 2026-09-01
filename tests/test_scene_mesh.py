"""The real scene reconstructed as geometry (tests/test_scene_mesh.py).

The reenactment is staged in the user's actual room, so the room has to BE
geometry - depth-tested against the hand and the object, lit by the same rig.
These pin the two things that make such a reconstruction either useful or
worthless: that it back-projects through the same pinhole as everything else,
and that it does not drape a surface across depth discontinuities.
"""

import numpy as np
import pytest

from src.simulation.render.scene_mesh import build_scene_mesh


def _frame(h=120, w=160):
    return np.full((h, w, 3), (40, 60, 90), np.uint8)


def test_a_flat_wall_reconstructs_at_its_own_depth():
    """A constant depth map is a plane at that distance. In lab world +Z points
    at the viewer, so a wall 1.5 m down the perception +Z axis sits at -1.5."""
    depth = np.full((120, 160), 1.5, np.float32)
    mesh = build_scene_mesh(depth, _frame(), grid_w=32)
    assert mesh is not None
    assert np.allclose(mesh.vertices[:, 2], -1.5, atol=1e-4)


def test_reconstruction_back_projects_through_the_shared_pinhole():
    """A vertex must land back on the pixel it was built from, under the
    fx = 0.8*w convention the rest of the system projects with. If it does not,
    the room is geometry in a different camera from the object standing on it.
    """
    h, w = 120, 160
    depth = np.full((h, w), 0.9, np.float32)
    mesh = build_scene_mesh(depth, _frame(h, w), grid_w=16)
    assert mesh is not None

    fx = 0.8 * w
    # Back to the perception frame, then project.
    cam = mesh.vertices * np.array([1.0, -1.0, -1.0], np.float32)
    u = fx * cam[:, 0] / cam[:, 2] + w / 2
    v = fx * cam[:, 1] / cam[:, 2] + h / 2
    assert u.min() >= -1 and u.max() <= w + 1, "reconstruction lands outside the frame"
    assert v.min() >= -1 and v.max() <= h + 1
    # The lattice spans the frame rather than collapsing to its centre.
    assert u.max() - u.min() > 0.8 * w
    assert v.max() - v.min() > 0.8 * h


def test_a_depth_cliff_is_left_as_a_hole_not_a_rubber_sheet():
    """Neighbouring pixels either side of an object's edge are metres apart in
    Z. Triangulating across them drapes a sheet from the foreground to the back
    wall, which is the single most conspicuous artefact of depth reconstruction
    and reads as a smear rather than a scene. The cell is dropped instead.
    """
    depth = np.full((120, 160), 2.5, np.float32)
    depth[:, :80] = 0.4                      # a hard cliff down the middle
    mesh = build_scene_mesh(depth, _frame(), grid_w=40)
    assert mesh is not None

    v = mesh.vertices
    spans = [abs(v[f, 2].max() - v[f, 2].min()) for f in mesh.faces]
    assert max(spans) < 0.5, (
        f"a triangle spans {max(spans):.2f} m of depth - the cliff was bridged")


def test_vertices_take_their_colour_from_the_frame():
    depth = np.full((120, 160), 1.0, np.float32)
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, :, 2] = 255                                  # pure red, BGR
    mesh = build_scene_mesh(depth, frame, grid_w=24)
    assert mesh is not None
    assert mesh.colors[:, 2].min() > 0.9 and mesh.colors[:, 0].max() < 0.1


@pytest.mark.parametrize("depth", [
    None,
    np.zeros((0, 0), np.float32),
    np.full((60, 80), np.nan, np.float32),
    np.full((60, 80), -1.0, np.float32),
])
def test_unusable_depth_yields_no_mesh(depth):
    """Depth drops out constantly. Returning an empty or nonsense mesh would
    put a sheet of garbage geometry in front of the camera."""
    assert build_scene_mesh(depth, _frame(60, 80), grid_w=16) is None
