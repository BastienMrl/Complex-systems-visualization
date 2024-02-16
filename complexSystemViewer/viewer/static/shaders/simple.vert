#version 300 es

//.... per vertex attributes ....
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;


//.... per mesh  attributes ....

layout(location = 3) in vec3 a_pos_t0;
layout(location = 4) in vec3 a_pos_t1;

layout(location = 5) in float a_state_t0;
layout(location = 6) in float a_state_t1;

// // color
// layout (location = 3) in vec3 a_color_t0;
// layout (location = 4) in vec3 a_color_t1;

// // transforms
// layout (location = 5) in vec3 a_translation_t0;
// layout (location = 6) in vec3 a_translation_t1;

// selection
layout (location = 10) in float a_selected;


out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out float v_selected;

uniform mat4 u_proj;
uniform mat4 u_view;

uniform float u_time_color;
uniform float u_time_translation;

mat4 create_translation_matrix(in vec3 translation){
    mat4 m;
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    m[2][2] = 1.0;
    m[3] = vec4(translation, 1.0);
    return m;
}


void factor_transformer(inout float value, const float factor, const float in_t0, const float in_t1, const float time){
    value = mix(in_t0, in_t1, time) * factor;
}




void interpolation_transformer(inout float value, const float factor_t0, const float factor_t1,
                                 const float in_t0, const float in_t1, const float time){
    float t = mix(in_t0, in_t1, time);
    value = mix(factor_t0, factor_t1, t);
}

void interpolation_transformer(inout vec3 value, const vec3 factor_t0, const vec3 factor_t1,
                                 const float in_t0, const float in_t1, const float time){
    float t = mix(in_t0, in_t1, time);
    value = mix(factor_t0, factor_t1, t);
}



void main(){
    vec3 translation = vec3(0., 0., 0.);
    const float factor_x = 0.95;
    const float factor_z = 0.95;
    const float factor_y = 1.5;
    factor_transformer(translation.x, factor_x, a_pos_t0.x, a_pos_t1.x, u_time_translation);
    factor_transformer(translation.z, factor_z, a_pos_t0.z, a_pos_t1.z, u_time_translation);
    factor_transformer(translation.y, factor_z, a_state_t0, a_state_t1, u_time_translation);

    const vec3 color_t0 = vec3(0.8705882352941177, 0.8901960784313725, 0.9294117647058824);
    const vec3 color_t1 = vec3(0.0392156862745098, 0.23137254901960785, 0.28627450980392155);
    interpolation_transformer(v_color, color_t0, color_t1, a_state_t0, a_state_t1, u_time_color);

    mat4 translation_matrix = create_translation_matrix(translation);
    mat4 transform = u_view * translation_matrix;
    vec4 view_pos = transform * vec4(a_position, 1.0);
    gl_Position = u_proj * view_pos;

    v_position = view_pos.xyz;
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
    v_uv = a_uv;
    v_selected = a_selected;
}