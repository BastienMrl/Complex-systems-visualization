import * as shaderUtils from "./shaderUtils.js"
import { vec3, mat4 } from "./glMatrix/esm/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";

class Viewer {
    gl;
    canvas;
    shaderProgram;
    resizeObserver;


    camera;
    #multipleInstances;

    #frameBuffer
    #selectionTargetTexture;
    #selectionDepthBuffer;
    #selectionProgram;

    #last_time = 0;

    mouseX;
    mouseY;
    #selectedId;

    #stats
    
    constructor(canvasId){
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext("webgl2");

        this.#stats = new Stats(document.getElementById("renderingFps"), document.getElementById("updateMs"), document.getElementById("renderingMs"));
    }
    
    async initialization(srcVs, srcFs, nbInstances){
        this.shaderProgram = await shaderUtils.initShaders(this.gl, srcVs, srcFs);
        this.gl.useProgram(this.shaderProgram);

        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0.2, 0.2, 0.2, 1.0);
        this.gl.enable(this.gl.CULL_FACE);
        this.gl.enable(this.gl.DEPTH_TEST);

        let self = this;
        
        await this.initSelectionBuffer("/static/shaders/selection.vert", "/static/shaders/selection.frag");
        this.initCamera(this.gl);
        this.initMesh(nbInstances);

        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);
    }

    async initMesh(nbInstances){
        
        this.#multipleInstances = new MultipleMeshInstances(this.gl, nbInstances);
        this.#multipleInstances.loadCube();

        let sqrtInstances = Math.sqrt(nbInstances);

        let offset = 2.1;
        let nbRow = sqrtInstances
        let offsetRow = vec3.fromValues(0, 0, offset);
        let offsetCol = vec3.fromValues(offset, 0, 0);
        let center = -(nbRow - 1) * offset / 2.;
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

    async initSelectionBuffer(srcVs, srcFs){
        this.gl.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();   
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });

        this.#selectionProgram = await shaderUtils.initShaders(this.gl, srcVs, srcFs);

        this.#frameBuffer = this.gl.createFramebuffer();
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.#frameBuffer);
        
        this.#selectionTargetTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.#selectionTargetTexture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.#selectionTargetTexture, 0);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        this.gl.bindTexture(this.gl.TEXTURE_2D, null);        
        
        this.#selectionDepthBuffer = this.gl.createRenderbuffer();
        this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, this.#selectionDepthBuffer);
        this.gl.framebufferRenderbuffer(this.gl.FRAMEBUFFER, this.gl.DEPTH_ATTACHMENT, this.gl.RENDERBUFFER, this.#selectionDepthBuffer);
        this.gl.renderbufferStorage(this.gl.RENDERBUFFER, this.gl.DEPTH_COMPONENT16, this.canvas.width, this.canvas.height);
        this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, null);

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    }

    onCanvasResize(){
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        const aspect = this.gl.canvas.clientWidth / this.gl.canvas.clientHeight;
        this.camera.setAspectRatio(aspect);
        this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

        this.gl.bindTexture(this.gl.TEXTURE_2D, this.#selectionTargetTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
       
        this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, this.#selectionDepthBuffer);
        this.gl.renderbufferStorage(this.gl.RENDERBUFFER, this.gl.DEPTH_COMPONENT16, this.canvas.width, this.canvas.height);
    }


    render(time){
        time *= 0.001;
        let delta = this.#last_time = 0 ? 0 : time - this.#last_time;
        this.#last_time = time

        this.#stats.startRenderingTimer(delta);
        this.#clear();
        this.#updateScene(delta);
        let selection = this.#getSelection();
        if (selection != this.#selectedId){
            this.#selectedId = (selection - 1) >= 0 ? (selection - 1) : null; 
            this.#multipleInstances.setMouseOver(this.#selectedId);
        }
        this.#draw();

       this.#stats.stopRenderingTimer();
    }

    updateState(data){
        this.#stats.startUpdateTimer();
        
        let colors = new Float32Array(data.length * 3);
        const c1 = [0.0392156862745098, 0.23137254901960785, 0.28627450980392155];
        const c2 = [0.8705882352941177, 0.8901960784313725, 0.9294117647058824];

        for (let i = 0; i < data.length; i++){
            for (let k = 0; k < 3; k++){
                colors[i * 3 + k] = c1[k] * data[i] + c2[k] * (1. - data[i]);
            }
        }
        this.#multipleInstances.updateColors(colors);

        this.#multipleInstances.updateYpos(data);

        this.#stats.stopUpdateTimer();
    }

    #updateScene(delta){
       //
    }

    #clear(){
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
    }

    #getSelection(){
        this.gl.useProgram(this.#selectionProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.#frameBuffer);
        this.#clear();

        var projLoc = this.gl.getUniformLocation(this.#selectionProgram, "u_proj");
        var viewLoc = this.gl.getUniformLocation(this.#selectionProgram, "u_view")


        this.gl.uniformMatrix4fv(projLoc, false, this.camera.getProjectionMatrix());
        this.gl.uniformMatrix4fv(viewLoc, false, this.camera.getViewMatrix());

        this.#multipleInstances.drawSelection();
        
        let data = new Uint8Array(4);
        this.gl.readPixels(this.mouseX, this.gl.canvas.height - this.mouseY, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, data)
        
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        return data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24);
    }

    #draw(){
        this.gl.useProgram(this.shaderProgram);

        var projLoc = this.gl.getUniformLocation(this.shaderProgram, "u_proj");
        var viewLoc = this.gl.getUniformLocation(this.shaderProgram, "u_view")
        var lightLoc = this.gl.getUniformLocation(this.shaderProgram, "u_light_loc");

        var lightPos = vec3.fromValues(0.0, 100.0, 10.0);
        vec3.transformMat4(lightPos, lightPos, this.camera.getViewMatrix());


        this.gl.uniformMatrix4fv(projLoc, false, this.camera.getProjectionMatrix());
        this.gl.uniformMatrix4fv(viewLoc, false, this.camera.getViewMatrix());

        this.gl.uniform3fv(lightLoc, lightPos);

        
        this.#multipleInstances.draw();
    }




}

export {Viewer}