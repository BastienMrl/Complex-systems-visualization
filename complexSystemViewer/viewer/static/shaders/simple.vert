#version 300 es

//.... per vertex attributes ....
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;


//.... per mesh  attributes ....

// color
layout (location = 3) in vec3 a_color_t0;
layout (location = 4) in vec3 a_color_t1;

// transforms
layout (location = 5) in vec3 a_translation_t0;
layout (location = 6) in vec3 a_translation_t1;

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

void main(){
    mat4 translation = create_translation_matrix(mix(a_translation_t0, a_translation_t1, u_time_translation));
    mat4 transform = u_view * translation;
    vec4 view_pos = transform * vec4(a_position, 1.0);
    gl_Position = u_proj * view_pos;

    v_position = view_pos.xyz;
    v_normal = transpose(inverse(mat3(transform))) * a_normal;
    v_color = mix(a_color_t0, a_color_t1, u_time_color);
    v_uv = a_uv;
    v_selected = a_selected;
}