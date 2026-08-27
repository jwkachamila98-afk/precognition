"""Session recording and telemetry logging utility (MP4 video + structured JSON)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SessionRecorder:
    """
    Session recording utility for robotic hand visuomotor telemetry.
    Saves synchronized MP4 compressed video frames alongside structured telemetry logs
    (MANO joint params, 112D state vectors, intent, residual actions, and rewards)
    under logs/sessions/YYYYMMDD_HHMMSS/.
    """

    def __init__(self, base_log_dir: str = "logs/sessions") -> None:
        self.base_log_dir = Path(base_log_dir)
        self.session_dir: Optional[Path] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.telemetry_file = None
        self._is_recording = False
        self._frame_count = 0
        self._session_metadata: Dict[str, Any] = {}

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start_recording(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        session_name: Optional[str] = None
    ) -> Path:
        """
        Initialize a new recording session folder and create video writer + telemetry stream.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = session_name or f"session_{timestamp_str}"
        self.session_dir = self.base_log_dir / folder_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        video_path = self.session_dir / "camera_feed.mp4"
        telemetry_path = self.session_dir / "session_telemetry.jsonl"

        # Initialize MP4 VideoWriter with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Open telemetry JSONL file
        self.telemetry_file = open(telemetry_path, "w", encoding="utf-8")

        self._is_recording = True
        self._frame_count = 0
        self._session_metadata = {
            "start_time": datetime.now().isoformat(),
            "resolution": [width, height],
            "fps": fps,
            "video_path": str(video_path),
            "telemetry_path": str(telemetry_path)
        }

        logger.info(f"Started recording session in: {self.session_dir}")
        return self.session_dir

    def record_frame(
        self,
        frame: np.ndarray,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single frame and its corresponding telemetry data record.
        """
        if not self._is_recording or self.video_writer is None:
            return

        # Write video frame
        self.video_writer.write(frame)
        self._frame_count += 1

        # Write telemetry JSON record
        if telemetry is not None and self.telemetry_file is not None:
            # Ensure numpy arrays are serialized to lists
            serializable_tel = self._sanitize_for_json(telemetry)
            serializable_tel["session_frame_idx"] = self._frame_count
            serializable_tel["record_time"] = datetime.now().isoformat()
            
            line = json.dumps(serializable_tel) + "\n"
            self.telemetry_file.write(line)
            self.telemetry_file.flush()

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert NumPy arrays and custom types to JSON-serializable primitives."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(x) for x in obj]
        return obj

    def stop_recording(self) -> Optional[Path]:
        """
        Finalize video writer, close telemetry stream, and save session summary manifest.
        """
        if not self._is_recording:
            return None

        self._is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.telemetry_file is not None:
            self.telemetry_file.close()
            self.telemetry_file = None

        # Write manifest summary JSON
        manifest_path = self.session_dir / "manifest.json"
        self._session_metadata["end_time"] = datetime.now().isoformat()
        self._session_metadata["total_frames"] = self._frame_count

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._session_metadata, f, indent=2)

        logger.info(f"Saved recording session ({self._frame_count} frames) to: {self.session_dir}")
        saved_dir = self.session_dir
        self.session_dir = None
        return saved_dir
