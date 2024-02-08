import * as shaderUtils from "./shaderUtils.js"
import { Vec3, Mat4 } from "./glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";

// provides access to gl constants
const gl = WebGL2RenderingContext


export class Viewer {
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    public shaderProgram : WebGLProgram;
    public resizeObserver : ResizeObserver;


    public camera : Camera;
    private _multipleInstances : MultipleMeshInstances;

    private _frameBuffer : WebGLFramebuffer | null;
    private _selectionTargetTexture : WebGLTexture | null;
    private _selectionDepthBuffer : WebGLRenderbuffer | null;
    private _selectionProgram : WebGLProgram;

    private _lastTime : number= 0;

    public mouseX : number;
    public mouseY : number;
    private _selectedId : number | null;

    private _stats : Stats;
    
    constructor(canvasId : string){
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        let context = this.canvas.getContext("webgl2");
        if (context == null){
            throw "Could not create WebGL2 context";
        }

        this.context = context;
        this._stats = new Stats(document.getElementById("renderingFps") as HTMLElement,
                                document.getElementById("updateMs") as HTMLElement,
                                document.getElementById("renderingMs") as HTMLElement);
    }
    
    public async initialization(srcVs : string, srcFs : string, nbInstances : number){
        this.shaderProgram = await shaderUtils.initShaders(this.context, srcVs, srcFs);
        this.context.useProgram(this.shaderProgram);

        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.enable(gl.DEPTH_TEST);

        
        await this.initSelectionBuffer("/static/shaders/selection.vert", "/static/shaders/selection.frag");
        this.initCamera();
        this.initMesh(nbInstances);
        
        let self = this;
        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);
    }

    private async initMesh(nbInstances : number){
        
        this._multipleInstances = new MultipleMeshInstances(this.context, nbInstances);
        this._multipleInstances.loadCube();

        let sqrtInstances = Math.sqrt(nbInstances);

        let offset = 2.05;
        let nbRow = sqrtInstances
        let offsetRow = Vec3.fromValues(0, 0, offset);
        let offsetCol = Vec3.fromValues(offset, 0, 0);
        let center = -(nbRow - 1) * offset / 2.;
        let firstPos = Vec3.fromValues(center, 0, center);
        this._multipleInstances.applyGridLayout(firstPos, sqrtInstances, sqrtInstances, offsetRow, offsetCol);
    }

    private initCamera(){
        const cameraPos = Vec3.fromValues(0., 80., 100.);
        const cameraTarget = Vec3.fromValues(0, 0, 0);
        const up = Vec3.fromValues(0, 1, 0);
    
        const fovy = Math.PI / 4;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        const near = 0.1;
        const far = 100000;
        
        
        this.camera = new Camera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
    }

    private async initSelectionBuffer(srcVs : string, srcFs : string){
        this.canvas.addEventListener('mousemove', (e : MouseEvent) => {
            const rect = this.canvas.getBoundingClientRect();   
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });

        this._selectionProgram = await shaderUtils.initShaders(this.context, srcVs, srcFs);

        this._frameBuffer = this.context.createFramebuffer();
        this.context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        
        this._selectionTargetTexture = this.context.createTexture();
        this.context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.context.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._selectionTargetTexture, 0);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        this.context.bindTexture(gl.TEXTURE_2D, null);        
        
        this._selectionDepthBuffer = this.context.createRenderbuffer();
        this.context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this.context.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this._selectionDepthBuffer);
        this.context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.canvas.width, this.canvas.height);
        this.context.bindRenderbuffer(gl.RENDERBUFFER, null);

        this.context.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    private onCanvasResize(){
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);

        this.context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
       
        this.context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this.context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.canvas.width, this.canvas.height);
    }


    public render(time : number){
        time *= 0.001;
        let delta = this._lastTime = 0 ? 0 : time - this._lastTime;
        this._lastTime = time

        this._stats.startRenderingTimer(delta);
        this.clear();
        this.updateScene(delta);
        let selection = this.getSelection();
        if (selection != this._selectedId){
            this._selectedId = (selection - 1) >= 0 ? (selection - 1) : null; 
            this._multipleInstances.setMouseOver(this._selectedId);
        }
        this.draw();

       this._stats.stopRenderingTimer();
    }

    public updateState(data){
        this._stats.startUpdateTimer();
        
        let colors = new Float32Array(data.length * 3);
        const c1 = [0.0392156862745098, 0.23137254901960785, 0.28627450980392155];
        const c2 = [0.8705882352941177, 0.8901960784313725, 0.9294117647058824];

        for (let i = 0; i < data.length; i++){
            for (let k = 0; k < 3; k++){
                colors[i * 3 + k] = c1[k] * data[i] + c2[k] * (1. - data[i]);
            }
        }
        this._multipleInstances.updateColors(colors);

        this._multipleInstances.updateYpos(data);

        this._stats.stopUpdateTimer();
    }

    private updateScene(delta){
       //
    }

    private clear(){
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }

    private getSelection(){
        this.context.useProgram(this._selectionProgram);
        this.context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        this.clear();

        var projLoc = this.context.getUniformLocation(this._selectionProgram, "u_proj");
        var viewLoc = this.context.getUniformLocation(this._selectionProgram, "u_view")


        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);

        this._multipleInstances.drawSelection();
        
        let data = new Uint8Array(4);
        this.context.readPixels(this.mouseX, this.context.canvas.height - this.mouseY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, data)
        
        this.context.bindFramebuffer(gl.FRAMEBUFFER, null);
        return data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24);
    }

    private draw(){
        this.context.useProgram(this.shaderProgram);

        var projLoc = this.context.getUniformLocation(this.shaderProgram, "u_proj");
        var viewLoc = this.context.getUniformLocation(this.shaderProgram, "u_view")
        var lightLoc = this.context.getUniformLocation(this.shaderProgram, "u_light_loc");

        var lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this.camera.viewMatrix);


        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);

        this.context.uniform3fv(lightLoc, lightPos);

        
        this._multipleInstances.draw();
    }




}
