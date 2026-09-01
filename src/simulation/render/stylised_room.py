"""A staging space that does not pretend to be a photograph.

Reconstructing the user's actual room from a single webcam was tried and does
not work: monocular depth is smooth, so a desk comes back as a relief, and no
amount of correct handling downstream repairs a soft input. Photographing the
room and compositing over it works, but then the shot is not geometry.

This is the third option. Authored geometry, so it always looks the way it was
designed to; REGISTERED, so it is placed at the real surface under the real
object as seen down the real camera axis. It reads as a clean 3-D staging of
the user's actual setup rather than a bad copy of their room - and because it
never claims to be a photograph, it cannot fall into the uncanny gap that made
the reconstruction look wrong.

The language is the app's own: near-black ground, one accent, no drawn edges,
falling off to nothing at the horizon rather than meeting a wall. The same
restraint the HUD uses, extended into three dimensions, so the reenactment
looks like part of this application instead of a visit to a different one.
"""

from __future__ import annotations

import numpy as np

from src.simulation.render.raster import Material, Mesh

# Near-black, very slightly cool, matching the HUD's card tint. Linear, since
# the shading pass works in linear light and tone-maps at the end.
_GROUND_BGR = (0.055, 0.050, 0.047)
# The accent, used only for the contact ring - the one place the eye should be.
_ACCENT_BGR = (1.00, 0.52, 0.04)

_GROUND_MATERIAL = Material(specular=0.10, shininess=18.0, cull_backfaces=False)


def build_stylised_room(
    surface_y: float,
    centre_xz: tuple = (0.0, 0.0),
    radius: float = 1.30,
    rings: int = 22,
    spokes: int = 40,
    contact_radius: float = 0.16,
) -> Mesh:
    """A ground plane under the object, fading to nothing at its edge.

    Built as a disc rather than a quad, and coloured so it darkens to black
    before it ends. A rectangular floor announces its own edges - two hard
    lines across the frame that say "this is a model" louder than anything
    standing on it - whereas a disc that fades out has no visible boundary at
    all, and needs no back wall to hide one.

    `surface_y` is the height of the real surface in LAB WORLD coordinates,
    normally the underside of the detected object, so the ground lands where
    the object is actually resting.
    """
    cx, cz = float(centre_xz[0]), float(centre_xz[1])

    # A fan of concentric rings: dense near the middle where the action is,
    # spreading out toward the fade.
    radii = np.linspace(0.0, 1.0, rings + 1) ** 1.6 * float(radius)
    angles = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)

    verts = [[cx, float(surface_y), cz]]
    for r in radii[1:]:
        for a in angles:
            verts.append([cx + r * float(np.cos(a)), float(surface_y),
                          cz + r * float(np.sin(a))])
    vertices = np.asarray(verts, dtype=np.float32)

    faces = []
    # Centre cap.
    for s in range(spokes):
        faces.append([0, 1 + s, 1 + (s + 1) % spokes])
    # Ring bands.
    for ring in range(len(radii) - 2):
        base = 1 + ring * spokes
        nxt = base + spokes
        for s in range(spokes):
            s1 = (s + 1) % spokes
            faces.append([base + s, nxt + s, base + s1])
            faces.append([base + s1, nxt + s, nxt + s1])
    faces = np.asarray(faces, dtype=np.int32)

    # Colour: the ground tint at the centre, falling to black at the rim, plus
    # a soft accent ring where the object meets the surface - the contact cue
    # that a shadow would give if there were a real floor to cast one onto.
    dist = np.linalg.norm(vertices[:, [0, 2]] - np.array([cx, cz], np.float32), axis=1)
    t = np.clip(dist / max(radius, 1e-6), 0.0, 1.0)
    fade = (1.0 - t) ** 2.2

    colors = np.asarray(_GROUND_BGR, np.float32)[None, :] * fade[:, None]
    if contact_radius > 1e-4:
        ring = np.exp(-((dist - contact_radius) / (contact_radius * 0.55)) ** 2)
        colors = colors + np.asarray(_ACCENT_BGR, np.float32)[None, :] * (
            0.16 * ring[:, None])

    normals = np.zeros_like(vertices)
    normals[:, 1] = 1.0                       # +Y is up in lab world

    return Mesh(vertices=vertices, faces=faces, normals=normals,
                colors=colors.astype(np.float32), material=_GROUND_MATERIAL)
