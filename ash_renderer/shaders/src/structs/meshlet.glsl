#extension GL_EXT_shader_8bit_storage : require

const uint MAX_PRIMITIVES = 126;
const uint MAX_VERTS = 64;
// https://github.com/KhronosGroup/Vulkan-Samples/blob/26eb94f79f9c2377cbaa28db2fbd7dd6efb55ffe/framework/api_vulkan_sample.h#L89
struct s_meshlet {
	uint vertices[MAX_VERTS];			 // 64
	uint8_t indices[4 * MAX_PRIMITIVES]; // 126 triangles, packed into bytes
	uint vertex_count;
	uint tri_count;
};
