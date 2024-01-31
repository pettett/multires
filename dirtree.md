```
.\
│
├── .config\
│   ├── config.toml
│   └── rustfmt.toml
│
├── ash_renderer\
│   │
│   ├── assets\
│   │   ├── .gitignore
│   │   ├── android.png
│   │   ├── apple.png
│   │   ├── linux.png
│   │   ├── vulkan.png
│   │   └── windows.png
│   │
│   ├── shaders\
│   │   │
│   │   ├── spv\
│   │   │   ├── compact_indices.comp
│   │   │   ├── frag_colour.frag
│   │   │   ├── frag_pbr.frag
│   │   │   ├── mesh-shader.frag
│   │   │   ├── mesh-shader.mesh
│   │   │   ├── mesh-shader.task
│   │   │   ├── mesh_shader_compute_cull.frag
│   │   │   ├── mesh_shader_compute_cull.mesh
│   │   │   ├── mesh_shader_compute_cull.task
│   │   │   ├── pbr_shader.frag
│   │   │   ├── should_draw.comp
│   │   │   └── vert.vert
│   │   │
│   │   └── src\
│   │       │
│   │       ├── structs\
│   │       │   ├── camera.glsl
│   │       │   ├── cluster_data.glsl
│   │       │   ├── indirect_draw_data.glsl
│   │       │   ├── mesh_frag_payload.glsl
│   │       │   ├── mesh_task_payload.glsl
│   │       │   ├── meshlet.glsl
│   │       │   ├── model.glsl
│   │       │   └── vertex.glsl
│   │       │
│   │       ├── clear_indirect_draw_data.comp
│   │       ├── compact_indices.comp
│   │       ├── frag_colour.frag
│   │       ├── frag_pbr.frag
│   │       ├── mesh-shader.mesh
│   │       ├── mesh-shader.task
│   │       ├── mesh_shader_compute_cull.task
│   │       ├── should_draw.comp
│   │       └── vert.vert
│   │
│   │
│   ├── src\
│   │   │
│   │   ├── app\
│   │   │   ├── fps_limiter.rs
│   │   │   ├── frame_measure.rs
│   │   │   ├── mod.rs
│   │   │   └── window.rs
│   │   │
│   │   ├── draw_pipelines\
│   │   │   ├── compute_culled_indices.rs
│   │   │   ├── compute_culled_mesh.rs
│   │   │   ├── draw_indirect.rs
│   │   │   ├── indirect_tasks.rs
│   │   │   ├── mod.rs
│   │   │   └── stub.rs
│   │   │
│   │   ├── gui\
│   │   │   ├── allocator_visualiser_window.rs
│   │   │   ├── gpu_allocator.rs
│   │   │   ├── gui.rs
│   │   │   ├── mod.rs
│   │   │   └── window.rs
│   │   │
│   │   ├── utility\
│   │   │   │
│   │   │   ├── pooled\
│   │   │   │   ├── command_pool.rs
│   │   │   │   ├── descriptor_pool.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── query_pool.rs
│   │   │   │
│   │   │   ├── buffer.rs
│   │   │   ├── constants.rs
│   │   │   ├── debug.rs
│   │   │   ├── device.rs
│   │   │   ├── image.rs
│   │   │   ├── instance.rs
│   │   │   ├── mod.rs
│   │   │   ├── physical_device.rs
│   │   │   ├── pipeline.rs
│   │   │   ├── platforms.rs
│   │   │   ├── render_pass.rs
│   │   │   ├── structures.rs
│   │   │   ├── surface.rs
│   │   │   ├── swapchain.rs
│   │   │   ├── sync.rs
│   │   │   └── tools.rs
│   │   │
│   │   ├── core.rs
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   ├── multires.rs
│   │   ├── screen.rs
│   │   └── vertex.rs
│   │
│   ├── Cargo.toml
│   ├── README.md
│   └── build.rs
│
├── assets\
│   ├── Duck.glb.bin
│   ├── chnd.glb.bin
│   ├── circle.glb.bin
│   ├── circle_low.glb.bin
│   ├── cone.glb.bin
│   ├── dragon_high.glb.bin
│   ├── maxwell.glb.bin
│   ├── monk.glb.bin
│   ├── monk_250k.glb.bin
│   ├── monk_60k.glb.bin
│   ├── monk_low.glb.bin
│   ├── plane.glb.bin
│   ├── plane_high.glb.bin
│   ├── pole.glb.bin
│   ├── sphere.glb.bin
│   ├── sphere_low.glb.bin
│   ├── torrin_main.glb.bin
│   ├── torus.glb.bin
│   └── torus_low.glb.bin
│
├── baker\
│   │
│   ├── src\
│   │   │
│   │   ├── mesh\
│   │   │   ├── edge.rs
│   │   │   ├── face.rs
│   │   │   ├── graph.rs
│   │   │   ├── group_info.rs
│   │   │   ├── mod.rs
│   │   │   ├── partition.rs
│   │   │   ├── plane.rs
│   │   │   ├── quadric.rs
│   │   │   ├── quadric_error.rs
│   │   │   ├── reduction.rs
│   │   │   ├── vertex.rs
│   │   │   └── winged_mesh.rs
│   │   │
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   └── pidge.rs
│   │
│   ├── tests\
│   │   └── tests.rs
│   │
│   ├── Cargo.toml
│   ├── asset.bin
│   ├── error.svg
│   ├── face_graph.svg
│   ├── face_graph2.svg
│   ├── graph.gv
│   ├── graph.svg
│   ├── hierarchy_graph.svg
│   └── part_graph.svg
│
├── common\
│   │
│   ├── src\
│   │   ├── asset.rs
│   │   ├── graph.rs
│   │   ├── lib.rs
│   │   ├── multi_res_mesh.rs
│   │   └── tri_mesh.rs
│   │
│   └── Cargo.toml
│
├── common_renderer\
│   │
│   ├── src\
│   │   │
│   │   ├── components\
│   │   │   ├── camera.rs
│   │   │   ├── camera_controller.rs
│   │   │   ├── gpu_mesh_util.rs
│   │   │   ├── mod.rs
│   │   │   └── transform.rs
│   │   │
│   │   ├── resources\
│   │   │   ├── mod.rs
│   │   │   └── time.rs
│   │   │
│   │   └── lib.rs
│   │
│   └── Cargo.toml
│
├── metis\
│   │
│   ├── src\
│   │   ├── lib.rs
│   │   └── partition.rs
│   │
│   ├── Cargo.toml
│   ├── LICENSE
│   ├── build.rs
│   ├── dot.gv
│   ├── manual-3.pdf
│   └── wrapper.h
│
├── svg\
│   ├── asset_dag.svg
│   ├── dot.gv
│   ├── error.svg
│   ├── error_partition_within_group.svg
│   ├── group_clustering_failure.svg
│   ├── group_int_clustering_failure.svg
│   ├── non_contig_graph.svg
│   ├── orig_dest_error.svg
│   ├── orig_dest_error_pre_collapse.svg
│   ├── triangle_plane.svg
│   ├── triangle_plane_row_divided.svg
│   ├── triangle_plane_unweighted_group.svg
│   └── triangle_plane_weighted_group.svg
│
│
├── .gitignore
├── .gitmodules
├── Cargo.lock
├── Cargo.toml
├── README.md
├── asset.bin
├── asset.chnd.bin
├── dot.gv
├── error.svg
├── error_partition_within_clusters.svg
├── error_partition_within_group.svg
├── flamegraph.svg
├── imgui.ini
└── vulkaninfo.txt
```
