#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;
layout (location = 3) in vec3 a_color;

layout (location = 4) in mat4 a_world;

out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;

uniform mat4 u_proj_view;


void main(){
    mat4 transform = u_proj_view * a_world;
    gl_Position = vec4(transform * vec4(a_position, 1.0));
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
    v_color = a_color;
    v_uv = a_uv;
}