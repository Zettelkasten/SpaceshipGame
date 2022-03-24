#version 330
in vec2 in_vert;

uniform mat3 world_transform;
uniform mat3 object_transform;

void main() {
    vec3 transformed_pos = world_transform * object_transform * vec3(in_vert, 1.0);
    gl_Position = vec4(transformed_pos.xy, 0.0, 1.0);
}
