
struct ClusterData {
    // Range into the index array that this submesh resides
    vec3 center;

    uint index_offset;
    uint index_count;

	float error;
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    int parent0;
    int parent1;

    int co_parent;
	// Filler draw_data for alignment
    float radius;
    uint layer;
    int max_child_index;

    int _4;
	int min_child_index;
    uint meshlet_start;
	uint meshlet_count;
};


struct DrawIndexedIndirect {
    uint index_count;
    uint instance_count;
    uint base_index;
    uint base_vertex;
    uint base_instance;
};
