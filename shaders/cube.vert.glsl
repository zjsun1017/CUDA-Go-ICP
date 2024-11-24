#version 330 core

in vec3 Position;
in vec3 Color;

out vec3 vertexColor;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

void main() {
    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(Position, 1.0);
    vertexColor = Color;
}
