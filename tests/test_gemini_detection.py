"""Open-vocabulary detection with tracking (tests/test_gemini_detection.py).

YOLO can only name the eighty COCO classes, so anything else went undetected or
was forced onto the nearest wrong label. Gemini names what is actually there,
but a call takes seconds and is billed, so it runs on a cadence with the boxes
tracked locally in between.
"""

import threading
import time

import numpy as np
import pytest

from src.perception.gemini_scene_detector import (DetectedObject,
                                                  GeminiObjectDetector,
                                                  GeminiSceneDetector)


def _detector(**kw):
    d = GeminiSceneDetector.__new__(GeminiSceneDetector)
    d.max_objects = kw.get("max_objects", 6)
    d.cadence_sec = kw.get("cadence_sec", 1.5)
    d._lock = threading.Lock()
    d._objects, d._trackers = [], {}
    d._inflight = False
    d._last_request_t = d._last_success_t = 0.0
    d._consecutive_failures = 0
    return d


def test_normalised_boxes_become_pixel_boxes():
    """Gemini returns [ymin, xmin, ymax, xmax] scaled 0-1000, not pixels."""
    d = _detector()
    objs = d._parse('[{"label":"coffee cup","box_2d":[100,200,400,500]}]', (480, 640))
    assert len(objs) == 1
    assert objs[0].box == (128, 48, 320, 192)   # x = 200/1000*640, y = 100/1000*480


@pytest.mark.parametrize("payload", [
    "not json at all", "[]", "{}", '[{"label":"x"}]', '[{"box_2d":[1,2,3,4]}]',
    '[{"label":"tiny","box_2d":[500,500,501,501]}]',
])
def test_malformed_or_degenerate_responses_are_discarded(payload):
    """A model will occasionally return prose, a missing field, or a box of no
    area. None of that may reach the scene."""
    assert _detector()._parse(payload, (480, 640)) == []


def test_boxes_are_clipped_to_the_frame():
    d = _detector()
    objs = d._parse('[{"label":"edge","box_2d":[-50,-50,1200,1200]}]', (480, 640))
    x1, y1, x2, y2 = objs[0].box
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480


def test_the_target_is_found_by_fuzzy_name():
    """The user says "cup"; the detector called it "coffee cup"."""
    d = _detector()
    d._objects = [DetectedObject("houseplant", (0, 0, 10, 10)),
                  DetectedObject("coffee cup", (20, 20, 60, 60))]
    assert d.find("cup").label == "coffee cup"
    assert d.find("coffee cup").label == "coffee cup"
    assert d.find("water cup").label == "coffee cup"
    assert d.find("banana") is None
    assert d.find("") is None


def test_tracking_carries_boxes_between_detections():
    """The whole point of the cadence: boxes stay glued at full frame rate while
    detection refreshes only every second or two."""
    d = _detector()
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[180:300, 250:390] = (200, 180, 90)
    d._adopt(frame, [DetectedObject("block", (250, 180, 390, 300))])
    assert d._trackers, "a tracker should have been seeded per object"

    before = d._objects[0].box
    for _ in range(5):
        d._advance_trackers(frame)
    after = d._objects[0].box
    assert max(abs(a - b) for a, b in zip(before, after)) < 12, "the box drifted off a static object"


def test_detection_is_never_requested_while_one_is_in_flight():
    """The render loop calls update() every frame; without this guard it would
    start dozens of billed requests a second."""
    d = _detector(cadence_sec=0.0)
    assert d._should_request() is True
    d._inflight = True
    assert d._should_request() is False


def test_update_never_blocks_and_always_returns_the_known_scene():
    d = _detector(cadence_sec=1e9)          # never due, so no network call
    d._objects = [DetectedObject("spoon", (10, 10, 50, 50))]
    frame = np.zeros((480, 640, 3), np.uint8)
    t0 = time.perf_counter()
    out = d.update(frame)
    assert (time.perf_counter() - t0) < 0.1, "update() must not wait on the network"
    assert [o.label for o in out] == ["spoon"]


def test_the_adapter_presents_detections_in_the_detector_interface():
    """So it can stand in wherever YOLO does."""
    from src.perception.object_detector import Detection2D
    a = GeminiObjectDetector.__new__(GeminiObjectDetector)
    a.scene = _detector(cadence_sec=1e9)
    a.scene._objects = [DetectedObject("utensil holder", (5, 6, 45, 46))]
    a.score_threshold, a._hint = 0.0, None
    out = a.detect(np.zeros((480, 640, 3), np.uint8))
    assert len(out) == 1 and isinstance(out[0], Detection2D)
    assert (out[0].label, out[0].xmin, out[0].ymax) == ("utensil holder", 5, 46)
