# Mesh Shader based renderer

## Instance work queue pseudocode

- We can rely solely on task shaders for each cluster, and repeat culling every frame.
   	- Doesn't work for meshes that can be both close and far

- We can allocate some shared memory per instance, and instead of having lots of workgroups, have a single workgroup output everything.

- We can have a compute pre-pass on the clusters for each instance, and draw those with the task shader in parallel.
   	- Extra compute shader
   	- Saves messing with grouping all the mesh shader outputs
   	- Init a queue with the lowest level of LOD, and an atomic counter. If any clusters fail the error check, increase the counter to allocate space for them, and place them into the queue at those points.
   	- Entire workgroup works in a loop, until there is complete agreement that the queue is empty.
   	- Successful clusters are placed onto a separate output buffer which is passed to the task shader for that instance
   	- Compute shader also writes the workgroup size to get the exact number of clusters to draw.
   	- This queue becomes the input to the next cycle
      		- This means most cycles are just copying data, which should be quite fast, but feels bad
   	- Intermediate memory required
      		- 4 bytes per (clusters/2) per mesh
