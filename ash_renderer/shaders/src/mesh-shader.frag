#version 460

#extension GL_ARB_separate_shader_objects: enable

layout (binding = 1) uniform sampler2D texSampler;

layout (location = 0) in Interpolants {
    vec3 fragColor;
    vec2 fragTexCoord;
} IN;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(IN.fragColor, 1);
}
