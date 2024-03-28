#extension GL_EXT_control_flow_attributes : require

// Calculates the frustum and planes from a projection matrix.
vec4[6] planes_from_mat(const mat4 mat) {
	vec4 planes[6];

	[[unroll]] for (int i = 0; i < 4; i++) { planes[0][i] = mat[i][3] + mat[i][0]; }
	[[unroll]] for (int i = 0; i < 4; i++) { planes[1][i] = mat[i][3] - mat[i][0]; }

	[[unroll]] for (int i = 0; i < 4; i++) { planes[2][i] = mat[i][3] + mat[i][1]; }
	[[unroll]] for (int i = 0; i < 4; i++) { planes[3][i] = mat[i][3] - mat[i][1]; }

	// Vulkan places it's near plane at w = 0, same as DX11, so use that part of the paper
	[[unroll]] for (int i = 0; i < 4; i++) { planes[4][i] = mat[i][3]; }
	[[unroll]] for (int i = 0; i < 4; i++) { planes[5][i] = mat[i][3] - mat[i][2]; }

	// Normalise planes
	[[unroll]] for (int i = 0; i < 6; i++) {
		float sqr_mag = dot(planes[i].xyz, planes[i].xyz);
		planes[i] *= inversesqrt(sqr_mag);
	}

	return planes;
}
// Calculate the distance from a plane to a point
float dist_to_plane(const vec4 plane, const vec3 point) {
	return dot(plane.xyz, point) + plane.w;
}
// Is a sphere within a set of six planes Based on the distance to each plane.
bool sphere_inside_planes(const vec4 planes[6], const vec4 sphere) {
	bool inside = true;
	[[unroll]] for (int i = 0; i < 6; i++) { inside = inside && (-sphere.w < dist_to_plane(planes[i], sphere.xyz)); }
	return inside;
}

// bool cluster_should_draw(uint cluster_index, uint instance_index, vec3 local_cam_pos) {
// 	mat4 MVP = ubo.culling_view_proj * models[instance_index].model;

// 	// Generate object space clipping planes
// 	vec4 planes[6];
// 	extract_planes_from_mat(MVP, planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]);
// #ifndef GROUP_SPHERE
// 	bool in_clip = sphere_inside_planes(planes, vec4(clusters[cluster_index].center, clusters[cluster_index].radius));
// #else
// 	bool in_clip = sphere_inside_planes(planes, clusters[cluster_index].tight_sphere);
// #endif

// 	if (dot(normalize(local_cam_pos - clusters[cluster_index].tight_sphere.xyz),
// 			clusters[cluster_index].tight_cone.xyz) < clusters[cluster_index].tight_cone.w) {
// 		in_clip = false;
// 	}
// 	return in_clip;
// }