import * as shaderUtils from "./shaderUtils.js"
import { Vec3, Mat4 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { TransformerBuilder } from "./transformerBuilder.js";
import { SelectionManager } from "./selectionTools/selectionManager.js";

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


    private _selectionManager : SelectionManager;

    private _lastTime : number= 0;


    private _stats : Stats;

    private _animationTimer : AnimationTimer;
    private _animationIds : [number, number];
    private _needAnimationPlayOnReceived : boolean = false;
    private _needOneAnimationLoop : boolean = false;


    private _transmissionWorker : Worker;
    private _currentValue : TransformableValues | null;
    private _nextValue : TransformableValues | null;

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

        this._selectionManager = new SelectionManager(this);
        
        this._currentValue = null;
        this._nextValue = null;
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
        
        await this.initCurrentVisu();
        this._drawable = true;
    }
    
    public async initCurrentVisu(){
        this._drawable = false;
        this._nextValue = null;
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.RESET);
        
        while (this._nextValue == null){
            await new Promise(resolve => setTimeout(resolve, 1));
        };

        this._currentValue = TransformableValues.fromInstance(this._nextValue);
        await this.initMesh(this._currentValue);
        this._drawable = true;
    }

    private async initMesh(values : TransformableValues){
        this._drawable = false;
        if (this._multipleInstances != null)
            delete this._multipleInstances;
        this._multipleInstances = new MultipleMeshInstances(this.context, values);
        this._selectionManager.setMeshes(this._multipleInstances);
        await this._multipleInstances.loadMesh("/static/models/cube_div_1.obj");
        this._drawable = true;
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





    public get selectionManager() : SelectionManager {
        return this._selectionManager;
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
        if (this._nextValue == null){
            return;
        }
        this._stats.startUpdateTimer();
        this._multipleInstances.updateStates(this._nextValue)
        this._currentValue = TransformableValues.fromInstance(this._nextValue);
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
            // let id = this._selectionManager.getMeshesId(this.mouseX, this.mouseY, this.canvas.width, this.canvas.height, this.camera);
            // this._multipleInstances.setMouseOver(id);
            this._stats.stopPickingTimer();
        }
        
        // rendering
        if (this._drawable){
            this._stats.startRenderingTimer(delta);
            this.clear();
            this.draw();
            this.context.finish();
            this._stats.stopRenderingTimer();
        }
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
        switch(getMessageHeader(e)){
            case WorkerMessage.READY:
                break;
            case WorkerMessage.VALUES_RESHAPED:
                this.onValuesReceived(getMessageBody(e), true);
                break;
            case WorkerMessage.VALUES:
                this.onValuesReceived(getMessageBody(e), false);
                break;
            case WorkerMessage.RESET:
                this.onReset();
                break;
        }
    }

    private async onReset(){
        this._nextValue = null;
        while (this._nextValue == null){
            await new Promise(resolve => setTimeout(resolve, 1));
        };
        this._multipleInstances.updateStates(this._nextValue);
        this._multipleInstances.updateStates(this._nextValue);

        this._currentValue = TransformableValues.fromInstance(this._nextValue);
    }

    public onValuesReceived(data : Array<Float32Array>, isReshaped : boolean = false){
        this._nextValue = TransformableValues.fromValuesAsArray(data);
        if (isReshaped){
            this._currentValue = this._nextValue;
            this.initMesh(this._nextValue)
        }
        if (!this._animationTimer.isRunning && this._needAnimationPlayOnReceived){
            this._needAnimationPlayOnReceived = false;
            this._needOneAnimationLoop = false;
            this.startVisualizationAnimation();
        }
        else if (!this._animationTimer.isRunning && this._needOneAnimationLoop){
            this._needOneAnimationLoop = false;
            this._multipleInstances.updateStates(this._nextValue);
            this.startOneAnimationLoop();
        }
    }

    public bindAnimationCurve(type : AnimableValue, fct : (time : number) => number){
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds[type] = id;
    }

    public startVisualizationAnimation() {
        if (this._animationTimer.isRunning){
            if (!this._animationTimer.loop)
                this._animationTimer.loop = true;
            return;
        }
        this.updateScene();
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }

    public stopVisualizationAnimation() {
        this._multipleInstances.updateStates(this._currentValue);
        this._animationTimer.stop();
    }

    public startOneAnimationLoop() {
        if (this._animationTimer.isRunning)
            return;
        this._animationTimer.loop = false
        // TODO set duration
        this._animationTimer.play();
    }

    public updateProgamsTransformers(transformers : TransformerBuilder){
        this.shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock());
    }

    public sendInteractionRequest(mask : Float32Array){
        if (this._animationTimer.isRunning && this._animationTimer.loop){
            this.stopVisualizationAnimation();
            this._needAnimationPlayOnReceived = true;
            this._multipleInstances.updateStates(this._currentValue);
        }
        else{
            this._needOneAnimationLoop = true;
        }
        let values = TransformableValues.fromInstance(this._currentValue);
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.APPLY_INTERACTION,
                            [mask].concat(values.toArray()), [mask.buffer].concat(values.toArrayBuffers()));
    }

}
