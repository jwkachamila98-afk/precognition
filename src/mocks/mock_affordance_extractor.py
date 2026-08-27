"""Synthetic Surface Affordance Grounding Mock."""

import math
import time
from typing import Optional
import numpy as np

from src.perception.scene_parser import BoundingBox3D
from src.simulation.trajectory_generator import AffordanceMap


class MockAffordanceExtractor:
    """
    Simulates a 3D visual affordance model (e.g. ContactArt / GraspNet / OpenMask3D).
    Computes surface contact probability distributions A(v_i) in [0, 1] over object geometry
    conditioned on task intent (e.g. grasp hotspots on a remote control body or handle).
    """

    def __init__(self, num_surface_points: int = 150) -> None:
        self.num_surface_points = num_surface_points

    def extract_affordance(
        self,
        bounding_box: BoundingBox3D,
        intent: str = "foresee me picking this remote control"
    ) -> AffordanceMap:
        """
        Generate surface contact probabilities and primary grasp hotspots for the target object.
        """
        center = bounding_box.center
        size = bounding_box.size
        label = bounding_box.label.lower()
        now = time.time()

        # Generate sample points on object box faces
        # Points along X, Y, Z extents
        dx, dy, dz = size / 2.0
        rand_pts = np.random.uniform(
            low=center - np.array([dx, dy, dz], dtype=np.float32),
            high=center + np.array([dx, dy, dz], dtype=np.float32),
            size=(self.num_surface_points, 3)
        ).astype(np.float32)

        # Contact hotspots calculation conditioned on intent and object type
        if "remote" in label or "remote" in intent.lower():
            # For remote control: pinch grasp hotspots on side rails and top surface
            # Hotspot 1: Thumb on left rail
            # Hotspot 2: Fingers on right rail
            # Hotspot 3: Palm resting behind
            h1 = center + np.array([-dx * 0.9, 0.0, 0.0], dtype=np.float32)
            h2 = center + np.array([ dx * 0.9, 0.0, 0.0], dtype=np.float32)
            h3 = center + np.array([ 0.0, -dy * 0.5, -dz * 0.8], dtype=np.float32)
            hotspots = np.array([h1, h2, h3], dtype=np.float32)
        elif "cup" in label or "mug" in label or "cup" in intent.lower():
            # Hotspot on cup handle and rim
            h1 = center + np.array([dx * 1.1, 0.0, 0.0], dtype=np.float32)
            h2 = center + np.array([dx * 0.6, -dy * 0.4, 0.0], dtype=np.float32)
            hotspots = np.array([h1, h2], dtype=np.float32)
        elif "bottle" in label or "bottle" in intent.lower():
            # Cylinder grasp around middle / neck
            h1 = center + np.array([-dx * 0.8, -dy * 0.2, 0.0], dtype=np.float32)
            h2 = center + np.array([ dx * 0.8, -dy * 0.2, 0.0], dtype=np.float32)
            hotspots = np.array([h1, h2], dtype=np.float32)
        else:
            # Generic opposed grasp hotspots
            h1 = center + np.array([-dx * 0.8, 0.0, 0.0], dtype=np.float32)
            h2 = center + np.array([ dx * 0.8, 0.0, 0.0], dtype=np.float32)
            hotspots = np.array([h1, h2], dtype=np.float32)

        # Compute distance from each surface point to nearest hotspot
        dists = np.min(np.linalg.norm(rand_pts[:, None, :] - hotspots[None, :, :], axis=-1), axis=-1)
        # Softmax / Gaussian probability: A(v_i) = exp(-d^2 / (2 * sigma^2))
        sigma = 0.04
        contact_probs = np.exp(- (dists ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)
        contact_probs = np.clip(contact_probs, 0.01, 1.0)

        return AffordanceMap(
            object_label=bounding_box.label,
            surface_points=rand_pts,
            contact_probabilities=contact_probs,
            hotspots=hotspots,
            intent=intent,
            timestamp=now
        )
