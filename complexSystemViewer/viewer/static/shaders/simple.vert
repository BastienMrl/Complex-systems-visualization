#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;
layout (location = 3) in vec3 a_color;

layout (location = 4) in mat4 a_world;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;

uniform mat4 u_proj;
uniform mat4 u_view;


void main(){
    mat4 transform = u_view * a_world;
    vec4 view_pos = transform * vec4(a_position, 1.0);
    gl_Position = u_proj * view_pos;
    v_position = view_pos.xyz;
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
    v_color = a_color;
    v_uv = a_uv;
}