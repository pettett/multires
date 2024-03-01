// This must be equal to or less than subgroup size

const uint TASK_GROUP_SIZE = 32;

const uint MAX_MESHLETS = 8;

struct MeshTaskPayload {
	uint instance;
	uint meshlet[TASK_GROUP_SIZE * MAX_MESHLETS];
	// uint meshlet_count[TASK_GROUP_SIZE];
};