import * as shaderUtils from "./shaderUtils.js"
import { Stats } from "./interface/stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { TransformerBuilder } from "./transformer/transformerBuilder.js";
import { UserInterface } from "./interface/userInterface.js";
import { Viewer } from "./viewer.js";
import { ViewerMultipleMeshes } from "./viewerMultipleMeshes.js";
import { ViewerTexture } from "./viewerTexture.js";

// provides access to gl constants
const gl = WebGL2RenderingContext

export enum ViewerType {
    MULTIPLE_MESHES = "Meshes",
    TEXTURE = "Texture"
}

export class ViewerManager {
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    public resizeObserver : ResizeObserver;

    private _viewers : Viewer[];
    private _currentViewer : Viewer;

    private _lastTime : number = 0;


    private _stats : Stats;


    public transmissionWorker : Worker;
    private _animationTimer : AnimationTimer
    private _animationIds : Map<shaderUtils.AnimableValue, number>;
    private _needAnimationPlayOnReceived : boolean = false;
    private _needOneAnimationLoop : boolean = false;

    private _values : TransformableValues | null;

    private _textures : TexturesContainer;


    constructor(canvasId : string){
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        let context = this.canvas.getContext("webgl2");
        if (context == null){
            throw "Could not create WebGL2 context";
        }
        this.context = context;

        this._animationTimer = new AnimationTimer(0.15, false);
        this._animationIds = new Map<shaderUtils.AnimableValue, number>();
        
        this._values = null;
        this.transmissionWorker = new Worker("/static/js/workers/transmissionWorker.js", {type : "module"});
        this.transmissionWorker.onmessage = this.onTransmissionWorkerMessage.bind(this);
        this._currentViewer = new ViewerMultipleMeshes(this.canvas, this.context, this);

        this._viewers = [this._currentViewer, new ViewerTexture(this.canvas, this.context, this)];
        this._textures = new TexturesContainer(this.context);
    }

    public set stats (stats : Stats){
        this._stats = stats;
    }
    

    public async initialization(viewer : ViewerType){
        this._viewers.forEach(viewer =>{
            viewer.initialization();
        });
        

        switch (viewer) {
            case ViewerType.MULTIPLE_MESHES:
                this._currentViewer = this._viewers[0];
                break;
            case ViewerType.TEXTURE:
                this._currentViewer = this._viewers[1];
                break;
        }
        let self = this;
        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);
        
        this._animationTimer.callback = function(){
            this.updateScene();
        }.bind(this);
        
        this._currentViewer.isDrawable = false;
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.RESET);
    }

    public switchViewer(viewer : ViewerType){
        switch (viewer) {
            case ViewerType.MULTIPLE_MESHES:
                this._currentViewer = this._viewers[0];
                break;
            case ViewerType.TEXTURE:
                this._currentViewer = this._viewers[1];
                break;
        }
    }





    // in seconds
    public set animationDuration(duration : number){
        this._animationTimer.duration = duration;
    }



    // private methods
    private onCanvasResize(){
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this._currentViewer.onCanvasResize();
    }
    
    private updateScene(){
        if (this._values == null){
            return;
        }
        this._stats.startUpdateTimer();
        this._values = null;
        this._textures.step();
        this.context.finish();
        this._stats.stopUpdateTimer();
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.GET_VALUES);
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
        
        
        // rendering
        if (this._currentViewer.isDrawable){
            this._stats.startRenderingTimer(delta);
            this._currentViewer.clear();
            this._currentViewer.draw(this._textures);
            this.context.finish();
            this._stats.stopRenderingTimer();
        }
    }

    public getAnimationTime(type : shaderUtils.AnimableValue){
        let id = this._animationIds.get(type);
        if (id == undefined || id == null)
            return this._animationTimer.getAnimationTime();
        return this._animationTimer.getAnimationTime(id);
    }

    private onTransmissionWorkerMessage(e : MessageEvent<any>){
        switch(getMessageHeader(e)){
            case WorkerMessage.READY:
                break;
            case WorkerMessage.VALUES_RESHAPED:
                this.onValuesReceived(getMessageBody(e), true);
                this._stats.logShape(this._values.nbElements, this._values.nbChannels);
                break;
            case WorkerMessage.VALUES:
                this.onValuesReceived(getMessageBody(e), false);
                break;
            case WorkerMessage.RESET:
                this.onReset();
                break;
            case WorkerMessage.SET_TIMER:
                this._stats.displayWorkerTimer(getMessageBody(e)[0], getMessageBody(e)[1])
                break;
        }
    }

    private async onReset(){
        this._values = null;
        while (this._values == null){
            await new Promise(resolve => setTimeout(resolve, 1));
        };
        this._textures.createBuffers(this._values);
        this._currentViewer.onReset(this._values);
    }

    public onValuesReceived(data : Array<Float32Array>, isReshaped : boolean = false){
        this._values = TransformableValues.fromValuesAsArray(data);
        if (isReshaped){
            let isChannels = this._values.nbChannels != this._textures.nbChannels;
            let isElements = this._values.nbElements != this._textures.nbElements;
            if (isChannels){
                UserInterface.getInstance().nbChannels = this._values.nbChannels;
                this._textures.createBuffers(this._values);
                this._currentViewer.onNbChannelsChanged(this._values)
            }
            if (isElements){
                this._textures.updateBuffers(this._values);
                this._currentViewer.onNbElementsChanged(this._values);
            }
        }
        else{
            this._textures.updateBuffers(this._values);
        }
        if (!this._animationTimer.isRunning && this._needAnimationPlayOnReceived){
            this._needAnimationPlayOnReceived = false;
            this._needOneAnimationLoop = false;
            this.startVisualizationAnimation();
        }
        else if (!this._animationTimer.isRunning && this._needOneAnimationLoop){
            this._needOneAnimationLoop = false;
            this._textures.updateBuffers(this._values);
            this.startOneAnimationLoop();
        }
    }

    public bindAnimationCurve(type : shaderUtils.AnimableValue, fct : (time : number) => number){
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds.set(type, id);
    }

    public startVisualizationAnimation() {
        if (this._animationTimer.isRunning){
            if (!this._animationTimer.loop)
                this._animationTimer.loop = true;
            return;
        }
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }

    public stopVisualizationAnimation() {
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
        this._currentViewer.updateProgamsTransformers(transformers);
    }

    public sendInteractionRequest(mask : Float32Array, interaction : string = "0"){
        if (this._animationTimer.isRunning && this._animationTimer.loop){
            this.stopVisualizationAnimation();
            this._needAnimationPlayOnReceived = true;
            this._textures.updateBuffers(this._values);
        }
        else{
            this._needOneAnimationLoop = true;
        }
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.APPLY_INTERACTION,
                            [interaction, this._textures.currentId, mask], [mask.buffer]);
    }

    public getElementOver(posX : number, posY : number) : number | null{
        return this._currentViewer.getElementOver(posX, posY);
    }
    
    public currentSelectionChanged(selection : Array<number> | null){
        this._currentViewer.currentSelectionChanged(selection);
    }


    public onMouseMoved(deltaX : number, deltaY : number){
        this._currentViewer.onMouseMoved(deltaX, deltaY);
    }

    public onWheelMoved(delta : number){
        this._currentViewer.onWheelMoved(delta);
    }

}

export class TexturesContainer{
    private _posXTexture : Array<WebGLTexture>;

    private _posYTexture : Array<WebGLTexture>;

    private _statesTextures : Array<Array<WebGLTexture>>;

    public context : WebGL2RenderingContext;

    private _step : 0 | 1 | 2 = 0;

    private _currentT : 0 | 1 | 2;

    private readonly _nbSteps = 3;

    private _nbElements : number;
    private _nbChannels : number;
    private _currentId : [number, number, number];

    public constructor(context : WebGL2RenderingContext){
        this.context = context;
        this._posXTexture = new Array<WebGLTexture>(this._nbSteps);
        this._posYTexture = new Array<WebGLTexture>(this._nbSteps);
        this._statesTextures = new Array<Array<WebGLTexture>>(this._nbSteps);
        this._currentId = [0, 0, 0];
    }

    public get nbElements() : number{
        return this._nbElements;
    }

    public get nbChannels() : number{
        return this._nbChannels;
    }

    public get currentId() : number{
        return this._currentId[this._step];
    }

    public getPosXTexture(t : 0 | 1){
        switch (this._step){
            case 0:
                return t == 0 ? this._posXTexture[0] : this._posXTexture[1];
            case 1:
                return t == 0 ? this._posXTexture[1] : this._posXTexture[2];
            case 2:
                return t == 0 ? this._posXTexture[2] : this._posXTexture[0];
        }
    }

    public getPosYTexture(t : 0 | 1){
        switch (this._step){
            case 0:
                return t == 0 ? this._posYTexture[0] : this._posYTexture[1];
            case 1:
                return t == 0 ? this._posYTexture[1] : this._posYTexture[2];
            case 2:
                return t == 0 ? this._posYTexture[2] : this._posYTexture[0];
        }
    }

    public getStatesTexture(t : 0 | 1){
        switch (this._step){
            case 0:
                return t == 0 ? this._statesTextures[0] : this._statesTextures[1];
            case 1:
                return t == 0 ? this._statesTextures[1] : this._statesTextures[2];
            case 2:
                return t == 0 ? this._statesTextures[2] : this._statesTextures[0];
        }
    }

    public createBuffers(values : TransformableValues){

        this._nbElements = values.nbElements;
        this._nbChannels = values.nbChannels;

        let width = Math.ceil(Math.sqrt(values.nbElements));
        let height = width;

        for (let i = 0; i < this._nbSteps; ++i){

            this._posXTexture[i] = this.context.createTexture();
            this.context.bindTexture(gl.TEXTURE_2D, this._posXTexture[i]);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            this._posYTexture[i] = this.context.createTexture();
            this.context.bindTexture(gl.TEXTURE_2D, this._posYTexture[i]);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            this._statesTextures[i] = new Array<WebGLTexture>(values.nbChannels);
            for (let k = 0; k < values.nbChannels; ++k){
                this._statesTextures[i][k] = this.context.createTexture();
                this.context.bindTexture(gl.TEXTURE_2D, this._statesTextures[i][k]);
                this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            }
        }

        this._currentT = 0;
        this.updateBuffers(values);
    }

    public updateBuffers(values : TransformableValues){

        this._nbElements = values.nbElements
        let width = Math.ceil(Math.sqrt(values.nbElements));
        let height = Math.floor(Math.sqrt(values.nbElements));
        let currentStep = (this._step + this._currentT) % this._nbSteps;
        this._currentId[currentStep] = values.id;
        console.log("id =", values.id, "updating step = ", currentStep)

        let fillEnd : boolean = false;
        if (width * height > this.nbElements)
            fillEnd = true;


        this.context.bindTexture(gl.TEXTURE_2D, this._posXTexture[currentStep]);
        let source = values.positionX;
        if (fillEnd){
            source = new Float32Array(width * height).fill(0., this.nbElements);
            source.set(values.positionX, 0)
        }
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);
        
        source = values.positionY
        this.context.bindTexture(gl.TEXTURE_2D, this._posYTexture[currentStep]);
        if (fillEnd){
            source = new Float32Array(width * height).fill(0., this.nbElements);
            source.set(values.positionY, 0)
        }
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);


        for (let i = 0; i < values.nbChannels; ++i){
            this.context.bindTexture(gl.TEXTURE_2D, this._statesTextures[currentStep][i]);
            source = values.states[i];
            if (fillEnd){
                source = new Float32Array(width * height).fill(0., this.nbElements);
                source.set(values.states[i], 0)
            }
            this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);

        }

        this._currentT += 1;
        if (this._currentT >= this._nbSteps)
            this._currentT = 2;
    }

    public step(){
        let before = this._step
        switch (this._step){
            case 0:
                this._step = 1;
                break
            case 1:
                this._step = 2;
                break;
            case 2:
                this._step = 0;
                break;
        }
        console.log("step before = ", before, "after = ",  this._step)
    }
}