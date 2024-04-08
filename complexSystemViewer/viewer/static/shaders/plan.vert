#version 300 es

//.... per vertex attributes ....
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;


out vec3 v_position;
out vec3 v_normal;
out vec2 v_uv;


uniform mat4 u_proj;
uniform mat4 u_view;



void main(){
    vec4 view_pos = u_view * vec4(a_position, 1.0);

    gl_Position = u_proj * view_pos;

    v_position = view_pos.xyz;
    v_normal = transpose(inverse(mat3(u_view))) * a_normal;
    v_uv = a_uv;
}