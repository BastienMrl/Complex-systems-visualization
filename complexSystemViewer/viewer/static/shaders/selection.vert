#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec4 a_id;

layout (location = 2) in mat4 a_world;

out vec4 v_id;

uniform mat4 u_proj;
uniform mat4 u_view;


void main(){
    v_id = a_id;

    mat4 transform = u_proj * u_view * a_world;
    gl_Position = transform * vec4(a_position, 1.0);
}