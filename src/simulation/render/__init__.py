"""Dependency-free software renderer for the simulated lab.

A z-buffered, perspective-correct triangle rasterizer with deferred shading,
built on numpy and cv2 alone - no GL context, no CUDA, no native 3-D
dependency - so it runs on the same CPU-only Intel Mac target as the rest of
the client.

Modules:
    camera      Perspective camera and the LAB WORLD / VIEW / SCREEN frames.
    raster      G-buffer rasterizer, Mesh/Material, winding utilities.
    shading     Deferred lighting, fog, and tone mapping.
    primitives  Procedural solids: box, cylinder, sphere, ring, tube, prism.
    textures    Procedural BGR textures for the lab surfaces.
    lab_scene   The static robotics lab: bench, backdrop, light rig, props.
    hand_mesh   Solid hand mesh built from the 21-joint keypoint skeleton.
    object_mesh Silhouette-inflated, photo-textured mesh of the real target.
"""
