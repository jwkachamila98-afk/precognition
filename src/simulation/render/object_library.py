"""Class-parametric meshes for the objects a hand picks up.

Silhouette inflation (object_mesh.py) makes no assumption about what the object
is, which is its virtue for anything unrecognised and its weakness for
everything else: a bottle comes out as a lumpy slab, because a single view
genuinely does not contain its depth.

But the detector tells us the CLASS, and almost everything on a bench is a
surface of revolution whose profile follows from that class. A bottle has a
body, a shoulder, a neck and a cap; a mug is a tapered cylinder with a floor.
Building the profile from the class and scaling it to the object's real size
produces correct geometry rather than inferred geometry - and it is the same
argument as the size prior in lab_sim: the class is better evidence than a
measurement taken through non-metric depth.

What is still real: the object's SIZE (its class prior) and its COLOUR, sampled
as a vertical ramp from the actual photo-crop so a green bottle stays green and
a dark cap stays dark. What is now assumed rather than observed: its shape,
which is a canonical example of its class, not this particular instance.
Unrecognised classes still go through silhouette inflation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.simulation.render import primitives as P
from src.simulation.render.raster import Material, Mesh

logger = logging.getLogger(__name__)

_MATERIAL = Material(specular=0.34, shininess=52.0)

# Profiles as (radius, height) in CENTIMETRES, bottom to top, at roughly life
# size for the class. Written in real units because that is the only way to
# sanity-check them by eye - an earlier version expressed radii as fractions of
# height and produced a "bottle" 19 cm across. Each profile is rescaled so that
# whichever of its height or diameter is LARGEST becomes the object's staged
# size, which also handles classes that are wider than they are tall (a bowl)
# without needing a separate case.
_PROFILES: dict = {
    # 500 ml water bottle: 7 cm across, 25 cm tall, shoulder into a narrow neck.
    "bottle": [(3.5, 0.0), (3.6, 1.0), (3.6, 15.0), (3.2, 18.0),
               (1.5, 20.5), (1.4, 23.0), (1.9, 23.4), (1.9, 25.0)],
    # Stemmed glass: wide foot, thin stem, tapered bowl.
    "wine glass": [(3.5, 0.0), (3.5, 0.4), (0.4, 1.2), (0.4, 8.0),
                   (2.6, 10.5), (4.0, 16.0), (4.0, 20.0)],
    "cup":  [(3.6, 0.0), (3.9, 0.5), (4.3, 8.8), (4.3, 9.5)],
    "mug":  [(3.8, 0.0), (4.1, 0.5), (4.3, 9.0), (4.3, 9.8)],
    "can":  [(2.9, 0.0), (3.3, 0.4), (3.3, 11.4), (2.9, 12.2)],
    "bowl": [(2.6, 0.0), (3.0, 0.5), (6.6, 5.4), (7.5, 7.0)],
    "vase": [(3.2, 0.0), (4.4, 2.5), (5.0, 13.0), (2.8, 22.0), (3.2, 25.0)],
    "apple":  [(0.0, 0.0), (2.7, 1.1), (3.9, 3.8), (2.9, 6.9), (0.0, 8.0)],
    "orange": [(0.0, 0.0), (2.7, 1.1), (4.0, 4.0), (2.7, 6.9), (0.0, 8.0)],
    "sports ball": [(0.0, 0.0), (6.6, 2.2), (11.0, 11.0), (6.6, 19.8), (0.0, 22.0)],
}

# Flat, boxy things - (width, height, depth) as fractions of the longest side.
_BOXES: dict = {
    "book": (1.0, 0.72, 0.16),
    "remote": (0.34, 1.0, 0.14),
    "cell phone": (0.50, 1.0, 0.07),
    "phone": (0.50, 1.0, 0.07),
    "mouse": (0.62, 1.0, 0.42),
    "keyboard": (1.0, 0.36, 0.05),
    "scissors": (0.42, 1.0, 0.06),
    "pen": (0.10, 1.0, 0.10),
    "stylus": (0.09, 1.0, 0.09),
    "toothbrush": (0.14, 1.0, 0.10),
    # Deliberately last-resort. Boxes run from matchbox to shipping carton, so
    # no single proportion is right for most of them - this is a shoebox, chosen
    # because it is graspable and unremarkable. It exists so an unrecognised
    # rectangular thing gets a rectangular mesh rather than an inflated blob.
    "box": (1.0, 0.55, 0.70),
}

# The upright axis of each profile is its height; a few classes are normally
# seen lying down, so their longest dimension is horizontal instead.
_LIES_DOWN = {"remote", "book", "keyboard", "mouse", "cell phone", "phone",
              "pen", "stylus", "scissors", "toothbrush"}


def _match(label: Optional[str], table: dict) -> Optional[str]:
    """Longest contained keyword wins, so 'wine glass' beats 'glass'."""
    if not label:
        return None
    text = label.replace("_", " ").strip().lower()
    for key in sorted(table, key=len, reverse=True):
        if key in text:
            return key
    return None


def colour_ramp(sprite: Optional[np.ndarray], levels: int = 12) -> Optional[np.ndarray]:
    """Bottom-to-top colour ramp sampled from the object's own photo-crop.

    The shape is now canonical, so colour is what keeps the staged object
    recognisable as the thing on the bench - a green bottle with a white label
    and a dark cap reads correctly even though the silhouette is generic.
    """
    if sprite is None or sprite.size == 0 or min(sprite.shape[:2]) < 4:
        return None
    small = cv2.resize(sprite, (8, levels), interpolation=cv2.INTER_AREA)
    # Median across the row resists the background bleeding in at the edges.
    ramp = np.median(small.astype(np.float32), axis=1) / 255.0
    return np.clip(ramp[::-1], 0.02, 1.0).astype(np.float32)     # image top = mesh top


def _apply_ramp(mesh: Mesh, ramp: Optional[np.ndarray]) -> Mesh:
    if ramp is None:
        return mesh
    y = mesh.vertices[:, 1]
    lo, hi = float(y.min()), float(y.max())
    t = np.clip((y - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    idx = t * (len(ramp) - 1)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(ramp) - 1)
    f = (idx - i0)[:, None]
    mesh.colors = (ramp[i0] * (1.0 - f) + ramp[i1] * f).astype(np.float32)
    return mesh


# Profiles authored for a specific object by GeminiMeshAuthor, keyed by label.
# Populated asynchronously; absent until an answer arrives, so the first staging
# of an object uses its canonical class shape and later ones use its own.
_AUTHORED: dict = {}


def remember_authored_profile(label: str, profile_cm) -> None:
    if label and profile_cm:
        _AUTHORED[label.strip().lower()] = [[float(r), float(h)] for r, h in profile_cm]


def authored_profile(label: Optional[str]):
    return _AUTHORED.get(label.strip().lower()) if label else None


def forget_authored_profiles() -> None:
    _AUTHORED.clear()


def build_class_mesh(label: Optional[str], longest_dim_m: float,
                     sprite: Optional[np.ndarray] = None,
                     segments: int = 20,
                     profile_cm=None) -> Optional[Mesh]:
    """Canonical mesh for a recognised class, centred on its own bounding box.

    Returns None when the class is unknown, so the caller falls back to
    silhouette inflation.
    """
    longest = max(float(longest_dim_m), 0.01)
    ramp = colour_ramp(sprite)
    base_colour = (0.62, 0.60, 0.58) if ramp is None else tuple(float(c) for c in ramp[len(ramp) // 2])

    if profile_cm:
        prof = np.asarray(profile_cm, dtype=np.float32).reshape(-1, 2)
        natural = max(float(prof[:, 1].max()), 2.0 * float(prof[:, 0].max()))
        prof *= longest / max(natural, 1e-6)
        mesh = P.lathe(prof, base_colour, segments=segments, material=_MATERIAL)
        mesh = _apply_ramp(mesh, ramp)
        v = mesh.vertices
        for axis in range(3):
            v[:, axis] -= 0.5 * (float(v[:, axis].min()) + float(v[:, axis].max()))
        logger.info(f"Object mesh: '{label}' built from a profile authored for this "
                    f"object ({len(prof)} points) at {longest*100:.0f} cm.")
        return mesh

    key = _match(label, _PROFILES)
    if key is not None:
        prof = np.asarray(_PROFILES[key], dtype=np.float32).copy()
        natural = max(float(prof[:, 1].max()), 2.0 * float(prof[:, 0].max()))
        prof *= longest / max(natural, 1e-6)
        mesh = P.lathe(prof, base_colour, segments=segments, material=_MATERIAL)
        mesh = _apply_ramp(mesh, ramp)
        logger.info(f"Object mesh: '{label}' modelled as a turned '{key}' profile "
                    f"at {longest*100:.0f} cm.")
    else:
        key = _match(label, _BOXES)
        if key is None:
            return None
        w, h, d = (np.asarray(_BOXES[key], dtype=np.float32) * longest)
        mesh = P.box((0.0, 0.0, 0.0), (w, h, d), base_colour, material=_MATERIAL)
        mesh = _apply_ramp(mesh, ramp)
        logger.info(f"Object mesh: '{label}' modelled as a '{key}' box "
                    f"at {longest*100:.0f} cm.")

    if key in _LIES_DOWN:
        # Lay it on the bench: its long axis becomes horizontal.
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        mesh = mesh.transformed(rotation=R)

    v = mesh.vertices
    for axis in range(3):
        v[:, axis] -= 0.5 * (float(v[:, axis].min()) + float(v[:, axis].max()))
    return mesh


def known_classes() -> List[str]:
    return sorted(set(_PROFILES) | set(_BOXES))
