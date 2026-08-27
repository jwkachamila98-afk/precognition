"""Real 2D object detection interfaces and a MediaPipe Tasks-backed implementation.

Unlike the language-grounded MockSceneParser (which places a canned bounding box based
solely on the intent string), this module runs an actual CPU object detector over the
live RGB frame so a detected bounding box corresponds to something really visible.
"""

import logging
import ssl
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import certifi
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# EfficientDet-Lite0 (COCO-80 classes), quantized int8 — ~4.5MB, runs comfortably on CPU.
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models" / "efficientdet_lite0.tflite"


@dataclass
class Detection2D:
    """A single real 2D object detection in pixel coordinates."""
    label: str
    score: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def center_px(self) -> Tuple[int, int]:
        return ((self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2)

    @property
    def width_px(self) -> int:
        return max(1, self.xmax - self.xmin)

    @property
    def height_px(self) -> int:
        return max(1, self.ymax - self.ymin)


class ObjectDetectorABC(ABC):
    """Abstract Base Class for real 2D object detectors running on RGB/BGR frames."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection2D]:
        """Detect objects in a camera frame (BGR, as produced by OpenCV), returning pixel-space boxes."""
        pass


class MediaPipeObjectDetector(ObjectDetectorABC):
    """
    Real-time CPU object detector using the MediaPipe Tasks API (EfficientDet-Lite0, COCO-80 classes).
    Downloads and caches the model on first use; the Tasks API is stable across both the
    legacy (0.10.x) and newer (1.x) mediapipe releases, unlike the removed `mp.solutions` API.
    """

    def __init__(self, score_threshold: float = 0.35, max_results: int = 5) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        model_path = self._ensure_model()
        options = vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            score_threshold=score_threshold,
            max_results=max_results,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)
        self._mp = mp

    @staticmethod
    def _ensure_model() -> Path:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _MODEL_PATH.exists() or _MODEL_PATH.stat().st_size < 1024:
            logger.info(f"Downloading EfficientDet-Lite0 object detection model to {_MODEL_PATH}...")
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(_MODEL_URL, context=ctx, timeout=30) as resp, open(_MODEL_PATH, "wb") as f:
                f.write(resp.read())
        return _MODEL_PATH

    def detect(self, image: np.ndarray) -> List[Detection2D]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(mp_image)

        h, w = image.shape[:2]
        detections: List[Detection2D] = []
        for det in result.detections:
            if not det.categories:
                continue
            category = det.categories[0]
            bbox = det.bounding_box
            detections.append(Detection2D(
                label=category.category_name or "object",
                score=float(category.score),
                xmin=int(np.clip(bbox.origin_x, 0, w - 1)),
                ymin=int(np.clip(bbox.origin_y, 0, h - 1)),
                xmax=int(np.clip(bbox.origin_x + bbox.width, 0, w)),
                ymax=int(np.clip(bbox.origin_y + bbox.height, 0, h)),
            ))
        return detections
