
struct ClusterData {
    // Range into the index array that this submesh resides
    index_offset: u32,
    index_count: u32,
    error: f32,
    //center: Vec3,
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    parent0: i32,
    parent1: i32,
    co_parent: i32,
}

@group(0) @binding(0) var<storage, read_write> draw: array<i32>;

@group(1) @binding(0) var<storage, read> clusters: array<ClusterData>;



@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;
    draw[i] = i32(clusters[i].index_offset);
}