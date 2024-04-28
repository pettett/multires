#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "structs\camera.glsl" // CameraUniformObject
#include "structs\mesh_frag_payload.glsl"
#include "structs\mesh_task_payload.glsl"
#include "structs\meshlet.glsl" // s_meshlet
#include "structs\model.glsl"	// Model
#include "structs\vertex.glsl"	// Vertex

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out Interpolants OUT;

layout(binding = 0) readonly buffer ModelUniformBufferObject {
	Model models[];
};

layout(binding = 3) readonly uniform CameraUniformBufferObject {
	CameraUniformObject ubo;
};

layout(location = 0) in vec4 pos;

layout(location = 1) in vec4 norm;

void main() {
	gl_Position = ubo.view_proj * models[gl_InstanceIndex].model * vec4(pos.xyz, 1);

	OUT.world_normal = (models[gl_InstanceIndex].model * vec4(norm.xyz, 0)).xyz;

	OUT.fragColor = vec3(0);

	OUT.fragTexCoord = vec2(0);
}
