import * as shaderUtils from "./shaderUtils.js"
import { Vec3, Mat4 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { StatesTransformer } from "./statesTransformer.js";
import { PickingTool } from "./pickingTool.js";

// provides access to gl constants
const gl = WebGL2RenderingContext

export enum AnimableValue {
    COULEUR = 0,
    POSITION = 1
}


export class Viewer {
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    public shaderProgram : shaderUtils.ProgramWithTransformer;
    public resizeObserver : ResizeObserver;


    public camera : Camera;
    private _multipleInstances : MultipleMeshInstances;


    private _pickingTool : PickingTool;

    private _lastTime : number= 0;


    private _stats : Stats;

    private _animationTimer : AnimationTimer;
    private _animationIds : [number, number];


    private _transmissionWorker : Worker;
    private _currentValue : TransformableValues | null;

    private _drawable : boolean;

    constructor(canvasId : string){
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        let context = this.canvas.getContext("webgl2");
        if (context == null){
            throw "Could not create WebGL2 context";
        }

        this.context = context;
        this._stats = new Stats(document.getElementById("renderingFps") as HTMLElement,
                                document.getElementById("updateMs") as HTMLElement,
                                document.getElementById("renderingMs") as HTMLElement,
                                document.getElementById("pickingMs") as HTMLElement,
                                document.getElementById("totalMs") as HTMLElement);

        this._animationTimer = new AnimationTimer(0.15, false);
        this._animationIds = [null, null];

        this._pickingTool = new PickingTool(this);
        
        this._currentValue = null;
        this._transmissionWorker = new Worker("/static/js/workers/transmissionWorker.js", {type : "module"});
        this._transmissionWorker.onmessage = this.onTransmissionWorkerMessage.bind(this);

        this._drawable = false;

        this.shaderProgram = new shaderUtils.ProgramWithTransformer(context);
    }
    
    // initialization methods
    public async initialization(srcVs : string, srcFs : string, nbInstances : number){
        await this.shaderProgram.generateProgram(srcVs, srcFs);
        this.context.useProgram(this.shaderProgram.program);
        
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK)
        this.context.enable(gl.DEPTH_TEST);
        
        
        this.initCamera();
        let self = this;
        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);
        
        this._animationTimer.callback = function(){
            this.updateScene();
        }.bind(this);
        
        await this.initCurrentVisu(nbInstances);
        this._drawable = true;
    }

    public async initCurrentVisu(nbElements : number){
        this._drawable = false;
        this._currentValue = null;
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.RESET, nbElements);
        
        while (this._currentValue == null){
            await new Promise(resolve => setTimeout(resolve, 1));
        };

        let values = this._currentValue;
        await this.initMesh(values);
        this._drawable = true;
    }

    private async initMesh(values : TransformableValues){
        if (this._multipleInstances != null)
            delete this._multipleInstances;
        this._multipleInstances = new MultipleMeshInstances(this.context, values);
        this._pickingTool.setMeshes(this._multipleInstances);
        await this._multipleInstances.loadMesh("/static/models/cube_div_1.obj");
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





    public get pickingTool() : PickingTool {
        return this._pickingTool;
    }

    public get transmissionWorker() : Worker {
        return this._transmissionWorker;
    }

    // setter

    // in seconds
    public set animationDuration(duration : number){
        this._animationTimer.duration = duration;
    }



    // private methods
    private onCanvasResize(){
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    
    private updateScene(){
        if (this._currentValue == null)
            return;
        this._stats.startUpdateTimer();
        this._multipleInstances.updateStates(this._currentValue)
        this.context.finish();
        this._stats.stopUpdateTimer();
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.GET_VALUES);
    }

    private clear(){
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }


    private draw(){
        this.context.useProgram(this.shaderProgram.program);

        let projLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_proj");
        let viewLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_view")
        let lightLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_light_loc");
        let timeColorLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_time_color");
        let timeTranslationLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_time_translation");
        let aabb = this.context.getUniformLocation(this.shaderProgram.program, "u_aabb");

        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this.camera.viewMatrix);


        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);
        
        this.context.uniform3fv(lightLoc, lightPos);
        
        this.context.uniform1f(timeColorLoc, this.getAnimationTime(AnimableValue.COULEUR));
        this.context.uniform1f(timeTranslationLoc, this.getAnimationTime(AnimableValue.POSITION));

        this.context.uniform2fv(aabb, this._multipleInstances.aabb, 0, 0);
        
        this._multipleInstances.draw();
    }

    public loopAnimation(){
        const loop = (time: number) => {
            this.render(time);
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }
    

    // public methods
    public render(time : number){
        time *= 0.001;
        let delta = this._lastTime = 0 ? 0 : time - this._lastTime;
        this._lastTime = time
        
        
        // picking
        if (this._drawable){
            this._stats.startPickingTimer();
            // let id = this._pickingTool.getMeshesId(this.mouseX, this.mouseY, this.canvas.width, this.canvas.height, this.camera);
            // this._multipleInstances.setMouseOver(id);
            this._stats.stopPickingTimer();
        }
        
        // rendering
        this._stats.startRenderingTimer(delta);
        this.clear();
        if (this._drawable)
            this.draw();
        this.context.finish();
        this._stats.stopRenderingTimer();
    }

    public currentSelectionChanged(selection : Array<number> | null){
        this._multipleInstances.updateMouseOverBuffer(selection);
    }

    private getAnimationTime(type : AnimableValue){
        let id = this._animationIds[type]
        if (id == null)
            return this._animationTimer.getAnimationTime();
        return this._animationTimer.getAnimationTime(id);
    }

    private onTransmissionWorkerMessage(e : MessageEvent<any>){
        switch(getMessageHeader(e.data)){
            case WorkerMessage.READY:
                break;
            case WorkerMessage.VALUES:
                let data = getMessageBody(e.data)
                this._currentValue = TransformableValues.fromValues(data[0], data[1]);
                break;
        }
    }

    public bindAnimationCurve(type : AnimableValue, fct : (time : number) => number){
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds[type] = id;
    }

    public startVisualizationAnimation() {
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }

    public stopVisualizationAnimation() {
        this._animationTimer.loop = false
    }

    public updateProgamsTransformers(transformers : StatesTransformer){
        this.shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock());
    }

}
