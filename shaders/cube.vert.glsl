#version 330 core

in vec4 Position;
in vec4 Color;

out vec4 vFragColorVs;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

void main() {
    // Pass the color to the geometry shader
    vFragColorVs = Color;

    // Apply model, view, and projection transformations
    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * Position;
}