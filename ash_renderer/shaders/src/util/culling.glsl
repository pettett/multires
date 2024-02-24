
void extract_planes_from_mat(
        const mat4 mat,
        out vec4 left,   out vec4 right,
        out vec4 bottom, out vec4 top,
        out vec4 near,   out vec4 far)
{
    for (int i = 4; i > 0; i--) { left[i]   = mat[i][3] + mat[i][0]; }
    for (int i = 4; i > 0; i--) { right[i]  = mat[i][3] - mat[i][0]; }
    for (int i = 4; i > 0; i--) { bottom[i] = mat[i][3] + mat[i][1]; }
    for (int i = 4; i > 0; i--) { top[i]    = mat[i][3] - mat[i][1]; }
    for (int i = 4; i > 0; i--) { near[i]   = mat[i][3] + mat[i][2]; }
    for (int i = 4; i > 0; i--) { far[i]    = mat[i][3] - mat[i][2]; }
}

float dist_to_plane(const vec4 plane, const vec3 point){
	return dot(plane.xyz, point) + plane.w;
}

bool sphere_inside_planes(const vec4 planes[6], const vec4 sphere){
	for (int i = 0; i < 6; i++){
		if (dist_to_plane(planes[i], sphere.xyz) + sphere.w * 2 < 0){
			return false;
		}
	}
	return true;
}
