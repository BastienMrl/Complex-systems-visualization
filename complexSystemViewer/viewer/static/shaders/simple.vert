#version 300 es

//.... per vertex attributes ....
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;


//.... per mesh  attributes ....

layout(location = 5) in vec3 a_translation_t0;
layout(location = 6) in vec3 a_translation_t1;

layout(location = 7) in float a_state_0_t0;
layout(location = 8) in float a_state_0_t1;

layout(location = 9) in float a_state_1_t0;
layout(location = 10) in float a_state_1_t1;

layout(location = 11) in float a_state_2_t0;
layout(location = 12) in float a_state_2_t1;

layout(location = 13) in float a_state_3_t0;
layout(location = 14) in float a_state_3_t1;


// selection
layout (location = 4) in float a_selected;


out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out float v_selected;

out vec3 feedback_translation;

uniform mat4 u_proj;
uniform mat4 u_view;

uniform float u_time_color;
uniform float u_time_translation;
uniform float u_time_rotation;
uniform float u_time_scaling;

uniform vec2 u_aabb[3];



mat4 create_transform_matrix(in vec3 translation, in vec3 rotation, in vec3 scaling){
    mat4 m;
    m[0][0] = cos(rotation.y) * cos(rotation.z) * scaling.x;
    m[0][1] = sin(rotation.x) * sin(rotation.y) * cos(rotation.z) - cos(rotation.x) * sin(rotation.z);
    m[0][2] = cos(rotation.x) * sin(rotation.y) * cos(rotation.z) + sin(rotation.x) * sin(rotation.z);

    m[1][0] = cos(rotation.y) * sin(rotation.z);
    m[1][1] = sin(rotation.x) * sin(rotation.y) * sin(rotation.z) + cos(rotation.x) * cos(rotation.z) * scaling.y;
    m[1][2] = cos(rotation.x) * sin(rotation.y) * sin(rotation.z) - sin(rotation.x) * cos(rotation.z);

    m[2][0] = -sin(rotation.y);
    m[2][1] = sin(rotation.x) * cos(rotation.y);
    m[2][2] = cos(rotation.x) * cos(rotation.y) * scaling.z;

    m[3] = vec4(translation, 1.0);
    return m;
}

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

void normalize_position(inout float value, const int idx){
    float min_value = u_aabb[idx][0];
    float max_value = u_aabb[idx][1];
    value = map(value, min_value, max_value, 0., 1.);
}



void main(){
    vec3 translation = vec3(0., 0., 0.);
    vec3 color = vec3(0., 0., 0.);
    vec3 rotation = vec3(0., 0., 0.);
    vec3 scaling = vec3(0., 0., 0.);

//${TRANSFORMERS}


    mat4 transformer_matrix = create_transform_matrix(translation, rotation, scaling);
    mat4 transform = u_view * transformer_matrix;
    vec4 view_pos = transform * vec4(a_position, 1.0);
    gl_Position = u_proj * view_pos;

    v_position = view_pos.xyz;
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
    v_color = color;
    v_uv = a_uv;
    v_selected = a_selected;
    feedback_translation = translation;
}