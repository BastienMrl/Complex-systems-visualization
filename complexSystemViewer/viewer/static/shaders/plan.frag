#version 300 es
precision mediump float;

in vec3 v_position;
in vec3 v_normal;
in vec2 v_uv;

layout (location = 0) out vec4 out_color;


uniform sampler2D tex_pos_x_t0;
uniform sampler2D tex_pos_y_t0;
uniform sampler2D tex_state_0_t0;
uniform sampler2D tex_selection;

uniform vec4 u_pos_domain;
uniform vec2 u_dimensions;



float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}


void factor_transformer(inout float value, const float factor, const float transformed_input){
    value += transformed_input * factor;
}


void interpolation_transformer(inout float value, const float factor_t0, const float factor_t1, const float transformed_input){
    value += mix(factor_t0, factor_t1, transformed_input);
}

void interpolation_transformer(inout vec3 value, const float factor_t0, const float factor_t1, const float transformed_input){
    value += mix(factor_t0, factor_t1, transformed_input);
}

void interpolation_transformer(inout vec3 value, const vec3 factor_t0, const vec3 factor_t1, const float transformed_input){
    value += mix(factor_t0, factor_t1, transformed_input);
}



void main(){
    vec3 color = vec3(0., 0., 0.);


    float x_coord = map(v_uv.x, 0., 1., 0., u_dimensions.x);
    float y_coord = map(v_uv.y, 0., 1., 0., u_dimensions.y);
    
    ivec2 tex_coord = ivec2(floor(x_coord), floor(y_coord));

//${TRANSFORMERS}

    float op = - sign(max(color.x, max(color.y, color.z)) - 0.5);
    vec3 selected = step(-0.5, texture(tex_selection, v_uv).r) * vec3(0.3, 0.3, 0.3);



    out_color = vec4(color + op * selected, 1.) ;
}