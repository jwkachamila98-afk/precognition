"""GPU-accelerated real object detector using Ultralytics YOLO.

Substantially more accurate than the CPU-tier EfficientDet-Lite0 MediaPipe detector,
and runs comfortably in real time on the RTX 4090 cloud GPU pod. Falls back to CPU
automatically if no CUDA device is present (e.g. local Intel Mac development).
"""

import logging
from typing import List

import numpy as np

from src.perception.object_detector import Detection2D, ObjectDetectorABC

logger = logging.getLogger(__name__)


class YoloObjectDetector(ObjectDetectorABC):
    """Real-time COCO-80 object detector backed by Ultralytics YOLO."""

    def __init__(self, model_name: str = "yolov8s.pt", conf_threshold: float = 0.35) -> None:
        import torch
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_name)
        self.model.to(self.device)
        logger.info(f"YoloObjectDetector: loaded {model_name} on device={self.device}")

    def detect(self, image: np.ndarray) -> List[Detection2D]:
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        h, w = image.shape[:2]
        names = results.names
        detections: List[Detection2D] = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0].cpu().numpy())
            label = names[int(box.cls[0].cpu().numpy())]
            detections.append(Detection2D(
                label=label,
                score=score,
                xmin=int(np.clip(xyxy[0], 0, w - 1)),
                ymin=int(np.clip(xyxy[1], 0, h - 1)),
                xmax=int(np.clip(xyxy[2], 0, w)),
                ymax=int(np.clip(xyxy[3], 0, h)),
            ))
        return detections
