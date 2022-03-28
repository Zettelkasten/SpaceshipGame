#version 330

uniform vec4 color;

out vec4 f_color;
void main() {
    // really exciting!
    f_color = color;
}
