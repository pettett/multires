// Camera uniform object. Likely updated every frame
struct CameraUniformObject {
    mat4 view_proj;
	mat4 culling_view_proj;
	vec3 camera_pos;
	float target_error;
};