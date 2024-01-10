
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
	// Filler draw_data for alignment
    radius: f32,
    _1: i32,
    _2: i32,

	_3: i32,
    _4: i32,
    _5: i32,
	_6: i32, 
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    base_index: u32,
    base_vertex: u32,
    base_instance: u32,
}

@group(0) @binding(0) var<storage, write> result_indicies: array<i32>;

@group(0) @binding(1) var<storage, write> draw_indirect_params: DrawIndexedIndirect;


@group(1) @binding(0) var<storage, read> indices: array<i32>;

@group(1) @binding(1) var<storage, read> clusters: array<ClusterData>;


@group(2) @binding(0) var<storage, read> should_draw: array<u32>;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;

    let cull = should_draw[i];

    if bool(cull) {
        let ind_start = clusters[i].index_offset;

        var result_start: u32 = u32(0);

		// Very slowly compact down the indices
        for (var meshlet: u32 = u32(0); meshlet < i; meshlet++) {
            result_start += should_draw[meshlet] * clusters[meshlet].index_count;
        }

		// Ideally, compute this in a step after writing cull to a buffer and compacting it down
        for (var ind: u32 = u32(0); ind < clusters[i].index_count; ind++) {
            result_indicies[result_start + ind] = indices[ind_start + ind] * i32(cull);
        }
    }

    if i32(i) == 0 {
        var total: u32 = u32(0);

		// Very slowly compact down the indices
        for (var meshlet: u32 = u32(0); meshlet < arrayLength(&clusters); meshlet++) {
            total += should_draw[meshlet] * clusters[meshlet].index_count;
        }

        draw_indirect_params.index_count = total;
    }
}