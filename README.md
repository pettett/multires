# Multiresolution Mesh Renderer

`Baker` converts meshes into a multiresolution asset, in this case a DAG of variable resolution clusters of triangles that serves as a basis for flexible view dependant LOD calculations.

Such a DAG may look like this:

![graph](baker/hierarchy_graph.svg)

Each level has its triangle count halved, and so each level contains half the clusters. Simplification takes place within groups, marked above with different coloured nodes, such that the boundaries of groups is not allowed to be changed. This allows an LOD renderer to substitute the sets of clusters within groups.

This format of mesh is perfect for the flexibility introduced by Mesh Shaders. `ash_renderer` is a Vulkan based multiresolution renderer capable of processing billions of source triangles to select LODs that keep screen space triangle density as close to a constant value as possible.

## TODO

### Ash-Renderer

- [ ] Compute culling stage (CCS)
   	- [ ] CCS -> Index compaction pipeline
   	- [x] CCS -> Mesh Shader output

- LOD viewer pipeline
- Instanced renderer pipeline

- Spacial locality for clusters in the same group

- Investigations:
   	- 2 phase occlusion culling
   	- Parallel work expansion techniques (using a queue in a compute shader)

## Diagrams

![flamegraph](flamegraph.svg)
