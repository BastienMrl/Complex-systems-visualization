import * as shaderUtils from "./shaderUtils.js"
import { vec3, mat4 } from "./glMatrix/esm/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";

class Viewer {
    gl;
    canvas;
    shaderProgram;

    camera;
    #multipleInstances;

    #last_time = 0;
    
    constructor(canvasId){
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext("webgl2");
    }
    
    async initialization(srcVs, srcFs, nbInstances){
        this.shaderProgram = await shaderUtils.initShaders(this.gl, srcVs, srcFs);

        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0.2, 0.2, 0.2, 1.0);
        this.gl.disable(this.gl.CULL_FACE);
        this.gl.enable(this.gl.DEPTH_TEST);

        this.initCamera(this.gl);
        this.initMesh(nbInstances);
    }

    async initMesh(nbInstances){
        
        this.#multipleInstances = new MultipleMeshInstances(this.gl, nbInstances);
        this.#multipleInstances.loadCube();

        let sqrtInstances = Math.sqrt(nbInstances);

        let offset = 2.5;
        let nbRow = sqrtInstances
        let offsetRow = vec3.fromValues(0, 0, offset);
        let offsetCol = vec3.fromValues(offset, 0, 0);
        let center = -nbRow * offset / 2.;
        let firstPos = vec3.fromValues(center, 0, center);
        this.#multipleInstances.applyGridLayout(firstPos, sqrtInstances, sqrtInstances, offsetRow, offsetCol);
    }

    initCamera(){
        const cameraPos = vec3.fromValues(0., 80., 100.);
        const cameraTarget = vec3.fromValues(0, 0, 0);
        const up = vec3.fromValues(0, 1, 0);
    
        const fovy = Math.PI / 4;
        const aspect = this.gl.canvas.clientWidth / this.gl.canvas.clientHeight;
        const near = 0.1;
        const far = 100000;
        
        
        this.camera = new Camera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
    }


    render(time){
        time *= 0.001;
        let delta = this.#last_time = 0 ? 0 : time - this.#last_time;
        this.#last_time = time

        this.#clear();
        this.#updateScene(delta);
        this.#draw();
    }

    updateState(data){
        let colors = new Float32Array(data.length * 3);
        const c1 = [0.55, 0.95, 0.65];
        const c2 = [0.5, 0.3, 0.7];

        for (let i = 0; i < data.length; i++){
            let c = data[i] ? c1 : c2;
            for (let k = 0; k < 3; k++){
                colors[i * 3 + k] = c[k];
            }
        }

        this.#multipleInstances.updateColors(colors);

        this.#multipleInstances.updateYpos(data);
    }

    #updateScene(delta){
       //
    }

    #clear(){
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    }

    #draw(){
        var projLoc = this.gl.getUniformLocation(this.shaderProgram, "u_proj");
        var viewLoc = this.gl.getUniformLocation(this.shaderProgram, "u_view")
        var lightLoc = this.gl.getUniformLocation(this.shaderProgram, "u_light_loc");

        var lightPos = vec3.fromValues(0.0, 100.0, 10.0);
        vec3.transformMat4(lightPos, lightPos, this.camera.getViewMatrix());
        console.log(lightPos)


        this.gl.uniformMatrix4fv(projLoc, false, this.camera.getProjectionMatrix());
        this.gl.uniformMatrix4fv(viewLoc, false, this.camera.getViewMatrix());

        this.gl.uniform3fv(lightLoc, lightPos);

        
        this.#multipleInstances.draw();
    }




}

export {Viewer}