
uint traverse_dag(uint idx, uint instance) {

	[[unroll]] for (int i = 0; i < STARTING; i++) { queue[i] = uint16_t(i); }

	uint queue_tail = STARTING;
	uint queue_head = 0;

	if (clusters.length() / 2 > QUEUE_SIZE) {
		while (true) {
		}
	}

	// leave some room for the number of clusters produced by the previous to increase
	// uint start = instance == 0 ? 0 : ranges[instance - 1].end + 100;

	// offset by the maximum cluster count in a mesh
	// uint start = instance * clusters.length() / 2;

	// if (subgroupElect()) {
	// 	ranges[instance].start = start;
	// }
	uint pos = 0;

	// if (subgroupElect()) {
	// 	debugPrintfEXT("Starting %v4f \n", planes[0]);
	// }

	// only evaluate if we are within the bounds of the queue
	while (queue_tail - queue_head > 0) {
		// old_queue_size = queue_tail - queue_head;

		subgroupMemoryBarrierShared();

		if (idx < queue_tail - queue_head) {
			queue_head += idx;
			uint i = queue[queue_head % QUEUE_SIZE];
			queue_head += 1;

			// should_draw[queue_head] = queue_head + 1;

			bool high_error;
			vec3 local_cam_pos;
			bool can_draw = cluster_can_draw(i, instance, local_cam_pos, high_error);

			// bool should_draw = sphere_inside_planes(planes, vec4(clusters[i].center, clusters[i].radius));

			vec4 local_pos = vec4(clusters[i].center, 1);

			// Generate object space clipping planes
			mat4 MVP = ubo.culling_view_proj * models[instance].model;

			vec4 planes[6] = planes_from_mat(MVP);

			bool should_draw_group = sphere_inside_planes(planes, vec4(clusters[i].center, clusters[i].radius));
			bool should_draw_cluster = sphere_inside_planes(planes, clusters[i].tight_sphere);

			// bool should_draw = cluster_should_draw(i, instance, local_cam_pos);

			// if (subgroupAll(draw)){
			// 	// Can just use idx instead of exclusive add for
			// simplicity
			// 	// Its pretty likely nearby clusters are drawn together
			// 	pos += idx;
			// }
			// else if (!subgroupAny(draw)){
			// 	// nothing
			// }
			// else if (draw){
			// 	pos += subgroupExclusiveAdd(1);
			// }

			if (can_draw && should_draw_cluster) {
				pos += subgroupExclusiveAdd(1);
				to_draw[pos] = uint16_t(i);
				pos += 1;
			}

			// pos += 1;

			bool can_queue_children = clusters[i].max_child_index >= STARTING;

			// add children to queue if we dont draw this and we have a
			// too high error

			if (i < clusters[i].co_parent && can_queue_children && should_draw_group && high_error) {
				// debugPrintfEXT("Min child index %d\n",
				// clusters[i].min_child_index); if
				// (clusters[i].min_child_index < i){ 	while
				// (true){}
				// }

				uint min_child_index = max(clusters[i].min_child_index, STARTING);

				// Allocate ourselves space at the end of the
				// queue. max_child_index >= STARTING from before

				uint children = clusters[i].max_child_index - min_child_index + 1;
				children = min(MAX_CHILDREN, children);

				queue_tail += subgroupExclusiveAdd(children);

				[[unroll]] for (uint c = 0; c < children; c++) {
					queue[(queue_tail + c) % QUEUE_SIZE] = uint16_t(min_child_index + c);
				}

				queue_tail += children;
			}
		}

		pos = subgroupMax(pos);
		queue_head = subgroupMax(queue_head);
		queue_tail = subgroupMax(queue_tail);

		// if (subgroupElect()){
		// 	debugPrintfEXT("\n");
		// }
	}
	return pos;
}

void output_selected_clusters(uint idx, uint instance, uint cluster_count, uint next_stage_group_size) {
	// if (subgroupElect()){ debugPrintfEXT("Finished \n"); }
	// if (subgroupElect()){ debugPrintfEXT("\n"); }
	uint start;
	if (subgroupElect()) {
		memoryBarrierBuffer();

		start = atomicAdd(ranges[0].end, cluster_count + next_stage_group_size);

		indirect_draw[instance].group_size_x = ((cluster_count + (next_stage_group_size - 1)) / next_stage_group_size);

		// debugPrintfEXT("Dispatch count: %u\n", cluster_count);

		ranges[instance + 1].start = start;
		ranges[instance + 1].end = start + cluster_count;
	}

	start = subgroupBroadcastFirst(start);

	// Copy to selected clusters list
	for (uint i = idx; i < cluster_count; i += LOCAL_SIZE_X) {
		draw_list[start + i] = to_draw[i] + 1;
	}
	if (idx < next_stage_group_size) {
		draw_list[start + cluster_count + idx] = 0;
	}
}
