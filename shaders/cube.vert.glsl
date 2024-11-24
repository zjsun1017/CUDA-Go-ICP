#version 330 core

in vec3 Position;
in int Flag;
in float Size;

out vec4 vertexColor;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

void main() {
    if (Flag == 0) {
        vertexColor = vec4(0.5, 0.5, 0.5, 1.0);
    } else if (Flag == 1) {
        vertexColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (Flag == 2) {
        vertexColor = vec4(1.0, 1.0, 0.0, 1.0);
    } else {
        vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
    }

    mat4 scaleMatrix = mat4(1.0);
    scaleMatrix[0][0] = Size;
    scaleMatrix[1][1] = Size;
    scaleMatrix[2][2] = Size;

    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * scaleMatrix * vec4(Position, 1.0);
}