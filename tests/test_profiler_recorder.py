"""Unit tests for Phase 5 LatencyProfiler and SessionRecorder."""

import json
import shutil
import time
from pathlib import Path
import numpy as np
import pytest

from src.utils.profiler import LatencyProfiler, StageStats
from src.utils.recorder import SessionRecorder


def test_latency_profiler():
    profiler = LatencyProfiler(window_size=50)

    # Test explicit start / stop
    profiler.start_stage("test_stage_1")
    time.sleep(0.005) # 5ms
    elapsed = profiler.stop_stage("test_stage_1")
    assert elapsed > 0.0

    # Test context manager
    for _ in range(10):
        with profiler.profile("test_stage_2"):
            time.sleep(0.001)

    stats_1 = profiler.get_stage_stats("test_stage_1")
    assert stats_1 is not None
    assert stats_1.sample_count == 1
    assert stats_1.mean_ms > 0.0

    stats_2 = profiler.get_stage_stats("test_stage_2")
    assert stats_2 is not None
    assert stats_2.sample_count == 10
    assert stats_2.p99_ms >= stats_2.min_ms

    table = profiler.format_table(fps=30.0)
    assert "test_stage_1" in table
    assert "test_stage_2" in table
    assert "TOTAL PIPELINE LATENCY" in table


def test_session_recorder(tmp_path: Path):
    rec_dir = tmp_path / "sessions"
    recorder = SessionRecorder(base_log_dir=str(rec_dir))

    assert not recorder.is_recording
    session_path = recorder.start_recording(width=100, height=100, fps=30, session_name="test_session")
    assert recorder.is_recording
    assert session_path.exists()

    # Record 5 test frames
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(5):
        telemetry = {
            "frame_id": i,
            "intent": "foresee me picking this remote control",
            "keypoints_3d": np.zeros((21, 3)).tolist(),
            "reward_score": 0.85,
            "discrepancy_norm": 0.02
        }
        recorder.record_frame(dummy_frame, telemetry)

    assert recorder.frame_count == 5

    saved_path = recorder.stop_recording()
    assert not recorder.is_recording
    assert saved_path is not None
    assert (saved_path / "camera_feed.mp4").exists()
    assert (saved_path / "session_telemetry.jsonl").exists()
    assert (saved_path / "manifest.json").exists()

    # Verify manifest JSON
    with open(saved_path / "manifest.json", "r") as f:
        manifest = json.load(f)
    assert manifest["total_frames"] == 5
    assert manifest["resolution"] == [100, 100]

    # Verify JSONL records
    with open(saved_path / "session_telemetry.jsonl", "r") as f:
        lines = f.readlines()
    assert len(lines) == 5
    first_record = json.loads(lines[0])
    assert first_record["frame_id"] == 0
    assert first_record["reward_score"] == 0.85
