

struct ClusterData {
	// Range into the index array that this submesh resides
	vec3 center;
	uint index_offset;

	// 4 bytes
	uint index_count;
	float error;
	// All of these could be none (-1), if we are a leaf or a root node
	int parent0;
	int parent1;

	// 4 bytes
	int co_parent;
	// Filler draw_data for alignment
	float radius;
	uint layer;
	int max_child_index;
	// 4 bytes
	int min_child_index;
	int _4;
	uint meshlet_start;
	uint meshlet_count;

	// 4 bytes
	vec4 tight_sphere;
	// 4 bytes
	vec4 tight_cone;
};

struct DrawIndexedIndirect {
	uint index_count;
	uint instance_count;
	uint base_index;
	uint base_vertex;
	uint base_instance;
};
