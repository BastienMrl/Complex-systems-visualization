#version 300 es

layout (location = 0) in vec3 a_position;
layout (location = 3) in vec4 a_id;

layout (location = 10) in vec3 a_translation_t0;
layout (location = 11) in vec3 a_translation_t1;

layout (location = 12) in float a_state_t0;
layout (location = 13) in float a_state_t1;


out vec4 v_id;

uniform mat4 u_proj_view;

uniform float u_time_translation;

void factor_transformer(inout float value, const float factor, const float transformed_input){
    value += transformed_input * factor;
}


void interpolation_transformer(inout float value, const float factor_t0, const float factor_t1, const float transformed_input){
    value += mix(factor_t0, factor_t1, transformed_input);
}

void interpolation_transformer(inout vec3 value, const vec3 factor_t0, const vec3 factor_t1, const float transformed_input){
    value += mix(factor_t0, factor_t1, transformed_input);
}


void main(){
    v_id = a_id;
    vec3 translation = vec3(0., 0., 0.);

//${TRANSFORMERS}

    vec3 position = a_position + translation;
    gl_Position = u_proj_view * vec4(position, 1.0);
}