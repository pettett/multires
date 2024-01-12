// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    part_highlight: i32,
};

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

@group(2) @binding(0) // 1.
var<uniform> model: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec4<f32>, 
    @location(1) normal: vec4<f32>, 
};

struct VertexOutput { 
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec4<f32>,
};


@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
	// 
    out.clip_position = camera.view_proj * (model * vec4<f32>(in.position.xyz, 1.0));
    out.world_normal = model * vec4<f32>(in.normal.xyz, 0.0);
    return out;
}
// Fragment shader
@fragment
fn fs_main(in: VertexOutput, @builtin(primitive_index) primitive_index: u32) -> @location(0) vec4<f32> {

    let l = normalize(vec3<f32>(0.5, 0.5, 0.5));
    let b = dot(normalize(in.world_normal.xyz), l);

    return vec4<f32>(b, b, b, 1.0);
}
