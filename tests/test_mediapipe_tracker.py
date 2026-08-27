"""Unit test verifying MediaPipeHandTracker compliance with HandTrackerABC."""

import numpy as np
import pytest
from src.perception.hand_tracker import HandTrackerABC
from src.perception.mediapipe_tracker import MediaPipeHandTracker, MEDIAPIPE_AVAILABLE


@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
def test_mediapipe_hand_tracker_interface():
    tracker = MediaPipeHandTracker(model_complexity=0)
    assert isinstance(tracker, HandTrackerABC)

    # Blank image should return 0 hand detections
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = tracker.estimate(dummy_img)
    assert isinstance(poses, list)
    assert len(poses) == 0
