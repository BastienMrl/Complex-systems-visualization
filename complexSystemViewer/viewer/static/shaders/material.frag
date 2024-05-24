#version 300 es
precision mediump float;
precision mediump sampler2DArray;

in vec3 v_position;
in vec3 v_normal;
in vec2 v_uv;

layout (location = 0) out vec4 out_color;


uniform sampler2DArray tex_t0;
uniform sampler2D tex_selection;

uniform samplerCube u_cube_map;

uniform vec3 u_light_loc;
uniform vec3 u_camera_loc;
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

vec4 get_color(in vec3 normal, in vec3 position, in vec3 color, in vec3 selected){
    vec3 light_dir = normalize(u_light_loc - position);
    float diffuse = (dot(normal, light_dir) * 0.5 + 0.5);
    float op = - sign(max(color.x, max(color.y, color.z)) - 0.5);

    vec3 h = (light_dir + vec3(0, 0, -1)) / 2.;

    float specular = pow(max(dot(h, normal), 0.), 1.5);
    
    return vec4(color * 0.1 + color * diffuse + op * selected + specular * vec3(0.9, 0.9, 0.9), 1);
}

vec4 get_normal_color(in vec3 normal){
    vec3 color = (normal + 1.) * 0.5;
    return vec4(color, 1.);
}

vec4 get_uv_color(in vec2 uv){
    return vec4(uv.x, uv.y, 0., 1.);
}

vec4 get_color_from_env_map(in vec3 normal, in vec3 light_dir, in vec3 color, in vec3 selected, in vec3 light_color, in vec3 camera_dir){
    float diffuse = (dot(normal, light_dir) * 0.5 + 0.5);
    float op = - sign(max(color.x, max(color.y, color.z)) - 0.5);

    vec3 h = (light_dir + camera_dir) / 2.;

    float specular = pow(max(dot(h, normal), 0.), 32.);

    vec3 c = vec3(color * 0.1);


    c += color * diffuse;
    c += specular * light_color;
    c += op * selected;

    return vec4(c, 1.);
}

void main(){
    vec3 color = vec3(0., 0., 0.);



    float x_coord = map(v_uv.x, 0., 1., 0., u_dimensions.x);
    float y_coord = map(v_uv.y, 0., 1., 0., u_dimensions.y);
    
    ivec2 tex_coord = ivec2(floor(x_coord), floor(y_coord));

//${TRANSFORMERS}

    float op = - sign(max(color.x, max(color.y, color.z)) - 0.5);
    vec3 selected = step(-0.5, texture(tex_selection, v_uv).r) * vec3(0.3, 0.3, 0.3);



    vec3 normal = normalize(v_normal);

    out_color = get_color(normal, v_position, color, selected);

    vec3 cam_to_surface = normalize(v_position - u_camera_loc);
    vec3 direction = reflect(cam_to_surface, v_normal);
 
    out_color = texture(u_cube_map, direction);

    out_color = get_color_from_env_map(normal, direction, color, selected, texture(u_cube_map, direction).rgb, -cam_to_surface);
        
}