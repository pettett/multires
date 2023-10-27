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
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position.xyz, 1.0);
    return out;
}
// Fragment shader
@fragment
fn fs_main(in: VertexOutput, @builtin(primitive_index) primitive_index: u32) -> @location(0) vec4<f32> {
    var part1 = partitions[primitive_index];
    var part2 = groups[primitive_index];
    var p = i32(primitive_index);
    var color1 = integer_to_rgb(&part1);
    var color2 = integer_to_rgb(&part2);

    return vec4<f32>(mix(color1, color2, 0.8), 1.0);
}
