#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;

out vec3 v_normal;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_world;


void main(){
    mat4 transform = u_projection * u_view * u_world;
    gl_Position = vec4(transform * vec4(a_position, 1.0));
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
}