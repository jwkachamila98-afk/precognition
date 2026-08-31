"""Non-blocking camera capture (src/utils/camera_stream.py).

`VideoCapture.read()` blocks until the sensor has a frame ready. At 30 fps that
is up to 33 ms of the render loop spent waiting on hardware - measured, on this
machine, as roughly a third of the entire frame budget, alongside ~35 ms in
`imshow` and ~34 ms of real work.

Reading on a background thread removes that wait from the loop entirely. It also
lowers latency: the reader keeps only the NEWEST frame and drops anything the
consumer was too slow to take, so the display always shows the present rather
than working through a queued backlog of stale frames.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np


class CameraStream:
    """Continuously drains a VideoCapture, exposing only the latest frame."""

    def __init__(self, capture, name: str = "camera") -> None:
        self._cap = capture
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._failures = 0
        self._running = True
        self._thread = threading.Thread(target=self._pump, name=name, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while self._running:
            cap = self._cap
            if cap is None:
                time.sleep(0.01)
                continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None and frame.size:
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._failures = 0
            else:
                with self._lock:
                    self._failures += 1
                # Back off a little on failure so a dead device does not spin a core.
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """The most recent frame. Never blocks on the sensor.

        Returns the same frame again if the consumer is running faster than the
        camera; callers that care can watch `sequence` to tell new from repeated.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    @property
    def sequence(self) -> int:
        """Increments once per frame actually delivered by the device."""
        with self._lock:
            return self._seq

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._failures

    def replace_capture(self, capture) -> None:
        """Swap the underlying device without restarting the thread."""
        with self._lock:
            self._cap = capture
            self._frame = None
            self._failures = 0

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
