import * as shaderUtils from "./shaderUtils.js"
import { vec3 } from "./glMatrix/esm/index.js";
import { Camera } from "./camera.js";
import { Mesh } from "./mesh.js";

class Viewer {
    gl;
    canvas;
    shaderProgram;

    #camera;
    #mesh;

    #last_time = 0;
    
    constructor(canvasId){
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext("webgl2");
    }
    
    async initialization(srcVs, srcFs){
        this.shaderProgram = await shaderUtils.initShaders(this.gl, srcVs, srcFs);

        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0.1, 0.1, 0.1, 1.0);
        this.gl.disable(this.gl.CULL_FACE);
        this.gl.enable(this.gl.DEPTH_TEST);

        this.initCamera(this.gl);
        this.initMesh();
    }

    async initMesh(){
        this.#mesh = new Mesh(this.gl);
        this.#mesh.loadCube();
    }

    initCamera(){
        const cameraPos = vec3.fromValues(0., 2., 4);
        const cameraTarget = vec3.fromValues(0, 0, 0);
        const up = vec3.fromValues(0, 1, 0);
    
        const fovy = Math.PI / 2;
        const aspect = this.gl.canvas.clientWidth / this.gl.canvas.clientHeight;
        const near = 0.1;
        const far = 100;
        
        
        this.#camera = new Camera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
    }


    render(time){
        time *= 0.001;
        let delta = this.#last_time = 0 ? 0 : time - this.#last_time;
        this.#last_time = time

        this.#clear();
        this.#updateScene(delta);
        this.#draw();
    }

    #updateScene(delta){
        this.#mesh.rotate(delta * Math.PI / 2, vec3.fromValues(0, 1, 0));
        this.#mesh.rotate(delta * Math.PI / 4, vec3.fromValues(0, 0, 1));
        this.#mesh.rotate(delta * Math.PI / 8, vec3.fromValues(1, 0, 0));
    }

    #clear(){
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    }

    #draw(){
        var projectionLoc = this.gl.getUniformLocation(this.shaderProgram, "u_projection");
        var viewLoc = this.gl.getUniformLocation(this.shaderProgram, "u_view");
        var worldLoc = this.gl.getUniformLocation(this.shaderProgram, "u_world");
        var lightLoc = this.gl.getUniformLocation(this.shaderProgram, "u_light_dir");
        var colorLoc = this.gl.getUniformLocation(this.shaderProgram, "u_color");

        this.gl.uniformMatrix4fv(projectionLoc, false, this.#camera.getProjectionMatrix());
        this.gl.uniformMatrix4fv(viewLoc, false, this.#camera.getViewMatrix());
        this.gl.uniformMatrix4fv(worldLoc, false, this.#mesh.getWorldMatrix());

        let lightdir = vec3.fromValues(0, 2, 4);
        vec3.normalize(lightdir, lightdir);
        this.gl.uniform3f(lightLoc, lightdir[0], lightdir[1], lightdir[2]);

        this.gl.uniform3f(colorLoc, 0, 0.5, 0.5);
        
        this.#mesh.draw();
    }




}

export {Viewer}