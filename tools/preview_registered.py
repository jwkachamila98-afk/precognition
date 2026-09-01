"""Offline preview of the REGISTERED lab (tools/preview_registered.py).

Renders the reenactment composited over a camera frame, and - the point of the
exercise - checks that the rendered object lands on the same pixels the live
overlay would draw its box on. Registration is a claim about pixels, so it is
verified in pixels rather than admired in a screenshot.

    PYTHONPATH=. python tools/preview_registered.py --out /tmp/registered.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.hand_tracker import HandPose, HandSide      # noqa: E402
from src.perception.scene_parser import BoundingBox3D            # noqa: E402
from src.simulation.lab_sim import LabSimulator                  # noqa: E402
from tools.preview_hud import fake_camera                        # noqa: E402
from tools.preview_overlay import _bbox, _recording, CAM_W, CAM_H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/registered.png")
    ap.add_argument("--step", type=float, default=0.72,
                    help="fraction through the reenactment")
    a = ap.parse_args()

    frame = fake_camera(CAM_W, CAM_H)
    bbox = _bbox()
    poses = _recording(bbox)

    sim = LabSimulator(width=CAM_W, height=CAM_H, registered=True)
    sprite = frame[150:250, 380:480].copy()
    if not sim.prepare_from_demonstration(poses, bbox, sprite):
        print("staging failed")
        return 1

    step = sim.step_for_progress(a.step)
    img = sim.render(step, elapsed=0.0, background=frame)
    if img is None:
        print("render returned None")
        return 1

    # Where the live overlay puts the object, from the same bbox.
    corners = bbox.project_to_2d(image_shape=(CAM_H, CAM_W))
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()

    # Where the RENDERED object actually landed. Measured at step 0, BEFORE
    # contact: past the grasp the demonstration carries the object away, so a
    # later step is supposed to disagree with the detected position.
    sim._rast.clear()
    sim._rast.draw(sim._object_mesh_for(0.0), sim.camera)
    cover = sim._rast.gbuffer[:, :, 12] > 0.5
    ys, xs = np.nonzero(cover)
    if len(xs) == 0:
        print("the object rendered no pixels at all")
        return 1
    rx0, ry0, rx1, ry1 = xs.min(), ys.min(), xs.max(), ys.max()

    print(f"live overlay box : x {x0:6.1f}..{x1:6.1f}   y {y0:6.1f}..{y1:6.1f}")
    print(f"rendered object  : x {rx0:6.1f}..{rx1:6.1f}   y {ry0:6.1f}..{ry1:6.1f}")
    dx = max(abs(rx0 - x0), abs(rx1 - x1))
    dy = max(abs(ry0 - y0), abs(ry1 - y1))
    print(f"disagreement     : {dx:.1f} px in x, {dy:.1f} px in y")

    # Draw the live box so the two can be compared by eye as well.
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 132, 10), 1)
    cv2.imwrite(a.out, img)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
