import * as shaderUtils from "./shaderUtils.js"
import { Vec3, Mat4 } from "./glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { StatesBuffer } from "./statesBuffer.js";
import { StatesTransformer, TransformType } from "./statesTransformer.js";
import { SelectionHandler } from "./selectionHandler.js";

// provides access to gl constants
const gl = WebGL2RenderingContext

export enum AnimableValue {
    COLOR = 0,
    TRANSLATION = 1
}


export class Viewer {
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    public shaderProgram : WebGLProgram;
    public resizeObserver : ResizeObserver;


    public camera : Camera;
    private _multipleInstances : MultipleMeshInstances;


    private _selectionHandler : SelectionHandler

    private _lastTime : number= 0;


    private _stats : Stats;

    private _animationTimer : AnimationTimer;
    private _animationIds : [number, number];


    private _statesBuffer : StatesBuffer;
    
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

        this._animationTimer = new AnimationTimer(1.5, true);
        this._selectionHandler = SelectionHandler.getInstance(context);
        this._animationIds = [null, null];
    }
    
    // initialization methods
    public async initialization(srcVs : string, srcFs : string, nbInstances : number){
        this.shaderProgram = await shaderUtils.initShaders(this.context, srcVs, srcFs);
        this.context.useProgram(this.shaderProgram);

        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.enable(gl.DEPTH_TEST);

        
        await this._selectionHandler.initialization("/static/shaders/selection.vert", "/static/shaders/selection.frag");
        this.initCamera();
        this.initMesh(nbInstances);
        
        let self = this;
        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);

        this._animationTimer.callback = function(){
            this.updateScene();
        }.bind(this);
        this._animationTimer.play();

        this._statesBuffer = new StatesBuffer(nbInstances, new StatesTransformer);
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


    // private methods
    private onCanvasResize(){
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);

        this._selectionHandler.resizeBuffers();
    }
    
    private updateScene(){
        let values = this._statesBuffer.values;
        this._multipleInstances.updateColors(values.colors);
        this._multipleInstances.updateTranslations(values.translations);
    }

    private clear(){
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }


    private draw(){
        this.context.useProgram(this.shaderProgram);

        let projLoc = this.context.getUniformLocation(this.shaderProgram, "u_proj");
        let viewLoc = this.context.getUniformLocation(this.shaderProgram, "u_view")
        let lightLoc = this.context.getUniformLocation(this.shaderProgram, "u_light_loc");
        let timeColorLoc = this.context.getUniformLocation(this.shaderProgram, "u_time_color");
        let timeTranslationLoc = this.context.getUniformLocation(this.shaderProgram, "u_time_translation");

        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this.camera.viewMatrix);


        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);
        
        this.context.uniform3fv(lightLoc, lightPos);
        
        this.context.uniform1f(timeColorLoc, this.getAnimationTime(AnimableValue.COLOR));
        this.context.uniform1f(timeTranslationLoc, this.getAnimationTime(AnimableValue.TRANSLATION));
        
        this._multipleInstances.draw();
    }
    

    // public methods
    public render(time : number){
        time *= 0.001;
        let delta = this._lastTime = 0 ? 0 : time - this._lastTime;
        this._lastTime = time
        
        this._stats.startRenderingTimer(delta);
        this.clear();


        // let prevSelection = this._selectionHandler.selectedId;
        // this._selectionHandler.updateCurrentSelection(this.camera, this._multipleInstances, this.getAnimationTime(AnimableValue.TRANSLATION));
        // let currentSelection = this._selectionHandler.selectedId;

        // if (this._selectionHandler.hasCurrentSelection() && currentSelection != prevSelection){
        //     this._multipleInstances.setMouseOver(currentSelection);
        // }
        
        this.draw();

       this._stats.stopRenderingTimer();
    }

    public updateState(data : any){
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

        // this._multipleInstances.updateTranslations(data);

        this._stats.stopUpdateTimer();
    }


    public setCurrentTransformer(transformer : StatesTransformer){
        this._statesBuffer.transformer = transformer;
    }

    public bindAnimationCurve(type : AnimableValue, fct : (time : number) => number){
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds[type] = id;
    }

    public getAnimationTime(type : AnimableValue){
        let id = this._animationIds[type]
        if (id == null)
            return this._animationTimer.getAnimationTime();
        return this._animationTimer.getAnimationTime(id);
    }

    public get statesBuffer() : StatesBuffer {
        return this._statesBuffer;
    }

}