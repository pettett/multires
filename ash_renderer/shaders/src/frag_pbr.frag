#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "structs\mesh_frag_payload.glsl"

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in Interpolants IN;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 l1 = normalize(vec3(0.5, 0.5, 0.5));
	vec3 l2 = normalize(vec3(-1.0, -0.5, 0.5));

	vec3 c1 = (vec3(0.8, 0.5, 0.5));
	vec3 c2 = (vec3(0.5, 0.5, 0.8));

	vec3 b = vec3(0.0);
	b += max(vec3(0.0), c1 * dot(normalize(IN.world_normal.xyz), l1));
	b += max(vec3(0.0), c2 * dot(normalize(IN.world_normal.xyz), l2));

	outColor = vec4(b, 1.0);
}
