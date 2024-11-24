#version 330

in vec3 vertPos;
in int Flag;

out vec4 fragColor;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

float hash(float seed) {
    return fract(sin(seed) * 43758.5453123);
}


void main() {
    if (Flag == 0) {
        fragColor = vec4(0.5, 0.5, 0.5, 1.0);
    } else if (Flag == 1) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (Flag == 2) {
        fragColor = vec4(1.0, 1.0, 0.0, 1.0);
    } else {
        fragColor = vec4(0.0, 1.0, 1.0, 1.0);
    }

    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(vertPos, 1.0);
}