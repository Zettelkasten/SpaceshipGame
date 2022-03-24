#version 330
in vec2 in_vert;

// per instance
in vec2 object_pos;
in float scale;

uniform mat3 world_transform;

void main() {
//    vec3 transformed_pos = world_transform * vec3(scale * in_vert + object_pos, 1.0);
    vec3 transformed_pos = world_transform * vec3(scale * in_vert + object_pos, 1.0);
    gl_Position = vec4(transformed_pos.xy, 0.0, 1.0);
}
