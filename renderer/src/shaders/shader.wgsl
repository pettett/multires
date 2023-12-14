// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    part_highlight: i32,
};

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> partitions: array<i32>;

@group(1) @binding(1)
var<storage, read> groups: array<i32>;

@group(2) @binding(0) // 1.
var<uniform> model: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec4<f32>, 
};

struct VertexOutput { 
    @builtin(position) clip_position: vec4<f32>, 
};

fn integer_to_rgb(integer: ptr<function, i32>) -> vec3<f32> {
    var red = f32((*integer * 109 + 47) % 269) / 269.0;
    var green = f32((*integer * 83 + 251) % 127) / 127.0;
    var blue = f32((*integer * 251 + 83) % 293) / 293.0;
    return vec3<f32>(red, green, blue);
}

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
	// 
    out.clip_position = camera.view_proj * (model * vec4<f32>(in.position.xyz, 1.0));
    return out;
}
// Fragment shader
@fragment
fn fs_main(in: VertexOutput, @builtin(primitive_index) primitive_index: u32) -> @location(0) vec4<f32> {
    var part = partitions[primitive_index];
    var group = groups[part];
    //var p = i32(primitive_index);
    var color1 = integer_to_rgb(&part);
    //var color2 = integer_to_rgb(&part2);

    return vec4<f32>(color1, 1.0);
}
