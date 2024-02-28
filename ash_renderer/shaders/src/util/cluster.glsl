

// Define INVOKE_PER_CLUSTER to say that workgroup count corresponds to maximum cluster being evaluated
// #define INVOKE_PER_CLUSTER

uint max_cluster() {
#ifdef INVOKE_PER_CLUSTER
	return min(clusters.length(), gl_NumWorkGroups.x * TASK_GROUP_SIZE);
#else
	return clusters.length();
#endif
}

float cluster_error(uint idx, vec3 local_cam_pos) {
	bool out_of_range = idx >= max_cluster();
	if (out_of_range) {
		return LARGE_ERROR;
	} else {
		vec3 center = clusters[idx].center;
		// vec3 cam =  ubo.camera_pos;
		// vec3 center = (models[idy].model * vec4(clusters[idx].center, 1.0)).xyz ;
		float radius = clusters[idx].radius;
		// float radius = length((models[idy].model * vec4(normalize(vec3(1)) * clusters[idx].radius, 0.0)).xyz);
		float error = clusters[idx].error;

		vec3 vec = center - local_cam_pos;
		float inv_center_distance = 1 / (dot(vec, vec));
		return (error * error * radius * radius) * inv_center_distance;
		// return clusters[idx].error;
	}
}

float get_parent_error(uint cluster_index, vec3 local_cam_pos) {
	// lots of zeros - clusters at the end are always avaliable to draw
	float parent_error = LARGE_ERROR;

	int parent0 = clusters[cluster_index].parent0;
	int parent1 = clusters[cluster_index].parent1;

	if (parent0 >= 0) { // If we have one parent, we are guarenteed to have another.
#ifdef MIN_ERROR
		parent_error = min(cluster_error(uint(parent0), local_cam_pos), cluster_error(uint(parent1), local_cam_pos));
#else
		parent_error = max(cluster_error(uint(parent0), local_cam_pos), cluster_error(uint(parent1), local_cam_pos));
#endif
	}
	return parent_error;
}

float get_this_error(uint cluster_index, vec3 local_cam_pos) {
	// lots of negative zeros - clusters at the base are always avaliable to draw
	float this_error = -LARGE_ERROR;

	int co_parent = clusters[cluster_index].co_parent;

	if (co_parent >= 0) {
#ifdef MIN_ERROR
		this_error = min(cluster_error(cluster_index, local_cam_pos), cluster_error(uint(co_parent), local_cam_pos));
#else
		this_error = max(cluster_error(cluster_index, local_cam_pos), cluster_error(uint(co_parent), local_cam_pos));
#endif
	}

	return this_error;
}

// // Calculate the largest possible index for the purpose of narrowing down the number of tasks dispatched.
// // May cause empty spots if the camera moves through more than 1 LOD per frame (highly unlikely)
// uint cacl_max_cluster_index(uint cluster_index, bool can_draw, float this_error, float error_cutoff){

// 	// (if not -1) parent0 < parent1 < cluster_index < max_child_index
// 	// (if root)   -1 <= -1 < cluster_index < max_child_index
// 	// (if leaf)   -1 < parent0 < parent1 < cluster_index
// 	return (can_draw ?
// 				cluster_index :  // Perfect error, continue doing this
// 				(error_cutoff < this_error ?
// 					max(clusters[cluster_index].max_child_index, cluster_index)  : // Error is lower then ours, draw a
// child if we have one 					clusters[cluster_index].parent1 == -1 ? cluster_index :
// clusters[cluster_index].parent1  // Error is higher then ours, only need to draw the parent (parent1 > parent0)
// 				)
// 			) ;

// }

bool cluster_can_draw(uint cluster_index, uint instance_index, out vec3 local_cam_pos, out bool high_error) {
	local_cam_pos = (models[instance_index].inv_model * vec4(ubo.camera_pos, 1.0)).xyz;
	// Calculate if this cluster should be drawn:
	// draw_error >= min(self.error, self.co-parent.error)
	// draw_error < min(parent0.error, parent1.error)
	float parent_error = get_parent_error(cluster_index, local_cam_pos);
	float this_error = get_this_error(cluster_index, local_cam_pos);

	float error = ubo.target_error * ubo.target_error * ubo.target_error;

	high_error = error < this_error;
	bool draw = error >= this_error && error < parent_error;

	return draw;
}
