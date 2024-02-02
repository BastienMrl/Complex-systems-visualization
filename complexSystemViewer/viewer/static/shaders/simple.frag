#version 300 es
precision mediump float;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;

out vec4 out_color;


uniform vec3 u_light_loc;


vec4 get_normal_color(in vec3 normal){
    vec3 color = (normal + 1.) * 0.5;
    return vec4(color, 1.);
}


vec4 get_color(in vec3 normal, in vec3 position){
    vec3 light_dir = normalize(u_light_loc - position);
    vec3 diffuse = v_color * (dot(normal, light_dir) * 0.5 + 0.5);
    return vec4(v_color * 0.4 + diffuse * 0.6, 1);
}

void main(){
    vec3 normal = normalize(v_normal);
    // out_color = get_normal_color(normal);
    out_color = get_color(normal, v_position);
}