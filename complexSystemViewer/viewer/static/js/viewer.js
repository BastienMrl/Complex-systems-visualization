import * as shaderUtils from "./shaderUtils.js"
import { vec3 } from "./glMatrix/esm/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";

class Viewer {
    gl;
    canvas;
    shaderProgram;

    #camera;
    #multipleInstances;

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
        this.initMesh(40 * 40);
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


        // test :
        for (let k = 0; k < 30; k++){
            let i = Math.floor(Math.random() * nbRow);
            let j = Math.floor(Math.random() * nbRow);
            let color = vec3.fromValues(Math.random(), Math.random(), Math.random());
            this.#multipleInstances.setColor(i, j, color);
        }

        this.#multipleInstances.updateColorBuffer();
    }

    initCamera(){
        const cameraPos = vec3.fromValues(0., 80., 100.);
        const cameraTarget = vec3.fromValues(0, 0, 0);
        const up = vec3.fromValues(0, 1, 0);
    
        const fovy = Math.PI / 4;
        const aspect = this.gl.canvas.clientWidth / this.gl.canvas.clientHeight;
        const near = 0.1;
        const far = 1000;
        
        
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
       //
    }

    #clear(){
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    }

    #draw(){
        var projViewLoc = this.gl.getUniformLocation(this.shaderProgram, "u_proj_view");
        var lightLoc = this.gl.getUniformLocation(this.shaderProgram, "u_light_dir");


        this.gl.uniformMatrix4fv(projViewLoc, false, this.#camera.getProjViewMatrix());

        let lightdir = vec3.fromValues(0, 2, 4);
        vec3.normalize(lightdir, lightdir);
        this.gl.uniform3f(lightLoc, lightdir[0], lightdir[1], lightdir[2]);

        
        this.#multipleInstances.draw();
    }




}

export {Viewer}