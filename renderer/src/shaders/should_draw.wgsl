
struct ClusterData {
    // Range into the index array that this submesh resides
    center: vec3<f32>,

    index_offset: u32,
    index_count: u32,

    error: f32,
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    parent0: i32,
    parent1: i32,

    co_parent: i32,
	// Filler data for alignment
    _0: i32,
    _1: i32,
    _2: i32,

	_3: i32,
    _4: i32,
    _5: i32,
	_6: i32, 
}

struct DrawData {
	camera_pos : vec3<f32>,
    error: f32,
	mode: i32,
}

@group(0) @binding(0) var<storage, write> draw: array<i32>;

@group(1) @binding(0) var<storage, read> clusters: array<ClusterData>;

@group(2) @binding(0) var<storage, read> indices: array<i32>;

@group(3) @binding(0) var<storage, read> data: DrawData;


fn cluster_error(idx: u32) -> f32{
	return clusters[idx].error / distance(clusters[idx].center, data.camera_pos);
}

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;
    // draw[i] = i32(clusters[i].index_offset);

    let start = clusters[i].index_offset;
    let end = start + clusters[i].index_count;

	// Calculate if this cluster should be drawn:
	// draw_error >= min(self.error, self.co-parent.error) 
	// draw_error < min(parent0.error, parent1.error)

	// lots of negative zeros
    var this_error = -100000000000000000.0;

    if clusters[i].co_parent >= 0 {
        this_error = min(cluster_error(i), cluster_error(u32(clusters[i].co_parent)));
    }

	// lots of zeros
    var parent_error = 100000000000000000.0;

    if clusters[i].parent0 >= 0 {
        parent_error = min(cluster_error(u32(clusters[i].parent0)), cluster_error(u32(clusters[i].parent1)));
    }

    let cull = i32(data.error >= this_error && data.error < parent_error);


	// Ideally, compute this in a step after writing cull to a buffer and compacting it down

    for (var ind: u32 = start; ind < end; ind++) {
        draw[ind] = indices[ind] * cull;
    }
}