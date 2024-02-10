#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec4 a_id;

layout (location = 5) in vec3 a_translation_t0;
layout (location = 6) in vec3 a_translation_t1;


out vec4 v_id;

uniform mat4 u_proj_view;

uniform float u_time;




void main(){
    v_id = a_id;

    vec3 translation = mix(a_translation_t0, a_translation_t1, u_time);
    vec3 position = a_position + translation;
    gl_Position = u_proj_view * vec4(position, 1.0);
}