# Multires

This repository is a proof of concept implementation of a multiresolution mesh rendering system following designs originally developed in _Batched Multi-Triangulation_ and _Nanite_.

It contains:

- Two simplification backends:
   	- Quadric error metrics (custom implementation, non uniform tessellation)
   	- `meshopt` (no support for simplifying boundaries)
- Two procedural geometry backends:
   	- Primitive shading (proof of concept, slow)
   	- Mesh Shading (blazing fast)
- Two cluster selection algroithms:
   	- DAG traverse - explore the DAG from the roots down, finding clusters that fit the error threshold.
   	- Adaptive Select - dispatch an invocation for each cluster that may be drawn (based on last frame). With mesh shading, this dispatches task shaders, with no intermediate memory usage.

More details to come when my dissertation is finished.
