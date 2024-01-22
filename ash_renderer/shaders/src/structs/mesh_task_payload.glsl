// This must be equal to or less than subgroup size

const uint TASK_GROUP_SIZE = 4;

struct MeshTaskPayload {
    uint instance;
    uint meshlet_start[TASK_GROUP_SIZE];
    uint meshlet_count[TASK_GROUP_SIZE];
};