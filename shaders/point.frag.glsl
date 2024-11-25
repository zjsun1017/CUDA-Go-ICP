#version 330

in vec4 vFragColor;
out vec4 fragColor;

void main() {
    if (length(vFragColor.rgb - vec3(0.0)) < 0.01) {
    discard;
}

    fragColor.r = abs(vFragColor.r);
    fragColor.g = abs(vFragColor.g);
    fragColor.b = abs(vFragColor.b);
}
