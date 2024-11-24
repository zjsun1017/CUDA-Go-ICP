#version 330

in vec4 Position;
in vec4 Color;
out vec4 vFragColorVs;

void main() {
    vFragColorVs = Color;
    gl_Position = Position;
}
