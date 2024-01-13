
struct ClusterData {
    // Range into the index array that this submesh resides
    center: vec3<f32>,

    index_offset: u32,
    index_count: u32,

    error: f32,
    //radius: f32,
    // All of these could be n1u (-1), if we are a leaf or a root node
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

//@group(0) @binding(2) var<storage, read_write> prefix_sum: array<u32>;

@group(1) @binding(0) var<storage, read> indices: array<i32>;

@group(1) @binding(1) var<storage, read> clusters: array<ClusterData>;

@group(2) @binding(0) var<storage, read> should_draw: array<u32>;

const pre_workgroup_summing_threads : u32 = 4u;
var<workgroup> workgroup_data: array<u32, pre_workgroup_summing_threads>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = id.x;
	// For the end workgroup
    if idx >= should_draw[0u] {
        return;
    }

    let cull = should_draw[idx + 1u];

	// Initialise scan to list of sizes [2,0,0,3,1,0,0]
    var idx_val = u32(cull) * clusters[idx].index_count;
    //prefix_sum[idx] = idx_val;

    //storageBarrier(); 
	// Very slowly compact down the indices

    let workgroup_start = idx - lid.x;

    if lid.x < pre_workgroup_summing_threads {

		// Sum everything before us in parallel
        for (var meshlet: u32 = u32(lid.x); meshlet < workgroup_start; meshlet += pre_workgroup_summing_threads) {
            workgroup_data[lid.x] += should_draw[meshlet + 1u] * clusters[meshlet].index_count;
        }
    }

    workgroupBarrier();

    var pre_workgroup_total = 0u;

    for (var i = u32(0); i < pre_workgroup_summing_threads; i++) {
        pre_workgroup_total += workgroup_data[i];
    }

    idx_val += pre_workgroup_total;

	// Add things within this workgroup
    for (var meshlet: u32 = workgroup_start; meshlet < idx; meshlet ++) {
        idx_val += should_draw[meshlet + 1u] * clusters[meshlet].index_count;
    }



    // for (var meshlet: u32 = u32(0); meshlet < idx; meshlet ++) {
    //     idx_val += should_draw[meshlet] * clusters[meshlet].index_count;
    // }



    // let len = arrayLength(&clusters);

    // for (var j = u32(1); j < len; j <<= u32(1)) {
    //     if idx + j < len {
    //         prefix_sum[idx + j] += prefix_sum[idx];
    //     }
    //     storageBarrier();
    // }


    //var offset = u32(0);
//     for (var i: u32 = u32(1); i < len; i <<= u32(1)) {

//         storageBarrier();
//         workgroupBarrier();

//         if (idx & i) != u32(0) && idx < len {
//             idx_val += prefix_sum[((idx >> offset) << offset) - u32(1)];
//         }

//         // if idx + i < len {
//         //     prefix_sum[idx + i] += prefix_sum[idx];
//         // }

//         storageBarrier();
//         workgroupBarrier();
// //
//         prefix_sum[idx] = idx_val;
// //
//         offset ++;
//     }

 
 //Upsweep
    // if idx < (len >> 1u) {
    //     prefix_sum[(idx << 1u) + 1u] += prefix_sum[idx << 1u];
    // }
    // var offset = u32(1);
    // for (var j = len >> u32(2); j > u32(0); j >>= 1u) {
    //     storageBarrier();
    //     workgroupBarrier();
    //     if idx < j {
    //         prefix_sum[(((idx << 1u) + u32(2)) << offset) - 1u] += prefix_sum[(((idx << 1u) + 1u) << offset) - 1u];
    //     }    offset++;
    // }
    // //Downsweep
    // for (var j = 1u; j < len; j <<= 1u) {
    //     offset--;
    //     storageBarrier();
    //     workgroupBarrier();
    //     if idx < j {
    //         prefix_sum[(((idx << 1u) + u32(3)) << offset) - 1u] += prefix_sum[(((idx << 1u) + u32(2)) << offset) - 1u];
    //     }
    // }





	// Scan should now look like [2,2,2,5,6,6,6]

	// Undo the starting addition to get our pack positions

	// Scan should now look like [0,-,-,2,5,-,-]
    if bool(cull) {

        idx_val -= clusters[idx].index_count;
        //var result_start: u32 = 0u;

        let ind_start = clusters[idx].index_offset;

		
		// Ideally, compute this in a step after writing cull to a buffer and compacting it down
        for (var ind: u32 = 0u; ind < clusters[idx].index_count; ind++) {
            result_indicies[idx_val + ind] = indices[ind_start + ind] * i32(cull);
        }

        idx_val += clusters[idx].index_count;
    }

	// Fill the max index count from the final work item

    if idx == should_draw[0u] - 1u {
        draw_indirect_params.index_count = idx_val;
    }
}