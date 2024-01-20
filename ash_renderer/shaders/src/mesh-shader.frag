#version 460

#extension GL_ARB_separate_shader_objects: enable

#extension GL_GOOGLE_include_directive : enable

#include "mesh_frag_payload.glsl"

layout (binding = 1) uniform sampler2D texSampler;

layout (location = 0) in Interpolants IN;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(IN.fragColor, 1);
}
