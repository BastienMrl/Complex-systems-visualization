#version 300 es
precision mediump float;

in vec4 v_id;

out vec4 out_color;

void main(){
    out_color = v_id;
}