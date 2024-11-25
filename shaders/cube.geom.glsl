#version 330 core

layout(points) in;
layout(points) out;
layout(max_vertices = 1) out;

in vec4 vFragColorVs[];
out vec4 vFragColor;

void main() {
    // Pass through the transformed position from the vertex shader
    gl_Position = gl_in[0].gl_Position;

    // Pass through the color
    vFragColor = vFragColorVs[0];

    EmitVertex();
    EndPrimitive();
}