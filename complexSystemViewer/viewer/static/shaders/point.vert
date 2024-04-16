#version 300 es

//.... per vertex attributes ....
layout (location = 0) in vec3 a_position;


out vec3 v_color;


uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec3 u_color;



void main(){
    vec4 view_pos = u_view * vec4(a_position, 1.0);

    gl_Position = u_proj * view_pos;
    v_color = u_color;
    gl_PointSize = 5.0;
}