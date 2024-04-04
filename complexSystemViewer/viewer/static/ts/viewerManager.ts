import * as shaderUtils from "./shaderUtils.js"
import { Stats } from "./interface/stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { TransformerBuilder } from "./transformer/transformerBuilder.js";
import { UserInterface } from "./interface/userInterface.js";
import { Viewer } from "./viewer.js";
import { ViewerMultipleMeshes } from "./viewerMultipleMeshes.js";

// provides access to gl constants
const gl = WebGL2RenderingContext

export enum ViewerType {
    MULTIPLE_MESHES,
    TEXTURE
}

export class ViewerManager {
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    public resizeObserver : ResizeObserver;

    private _viewers : Viewer[];
    private _currentViewer : Viewer;

    private _lastTime : number= 0;


    private _stats : Stats;


    public transmissionWorker : Worker;
    private _animationTimer : AnimationTimer
    private _animationIds : Map<shaderUtils.AnimableValue, number>;
    private _needAnimationPlayOnReceived : boolean = false;
    private _needOneAnimationLoop : boolean = false;

    private _currentValue : TransformableValues | null;
    private _nextValue : TransformableValues | null;

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
        
        this._currentValue = null;
        this._nextValue = null;
        this.transmissionWorker = new Worker("/static/js/workers/transmissionWorker.js", {type : "module"});
        this.transmissionWorker.onmessage = this.onTransmissionWorkerMessage.bind(this);
        this._currentViewer = new ViewerMultipleMeshes(this.canvas, this.context, this);

        this._viewers = [this._currentViewer];
        this._textures = new TexturesContainer()
    }

    public set stats (stats : Stats){
        this._stats = stats;
    }
    

    public async initialization(viewer : ViewerType){
        switch (viewer) {
            case ViewerType.MULTIPLE_MESHES:
                this._currentViewer = this._viewers[0];
            case ViewerType.TEXTURE:
                break;
        }
        this._currentViewer.initialization();
        let self = this;
        this.resizeObserver = new ResizeObserver(function() {self.onCanvasResize();});
        this.resizeObserver.observe(this.canvas);
        
        this._animationTimer.callback = function(){
            this.updateScene();
        }.bind(this);
        
        this._currentViewer.isDrawable = false;
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.RESET);
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
        if (this._nextValue == null){
            return;
        }
        this._stats.startUpdateTimer();
        this._currentValue = this._nextValue;
        this._nextValue = null;
        this._currentViewer.updateScene(this._currentValue);
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
            this._currentViewer.draw();
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
                this._stats.logShape(this._currentValue.nbElements, this._currentValue.nbChannels);
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
        this._nextValue = null;
        while (this._nextValue == null){
            await new Promise(resolve => setTimeout(resolve, 1));
        };
        this._currentValue = TransformableValues.fromInstance(this._nextValue);
        this._currentViewer.onReset(this._currentValue);
    }

    public async onValuesReceived(data : Array<Float32Array>, isReshaped : boolean = false){
        this._nextValue = TransformableValues.fromValuesAsArray(data);
        this._textures.createBuffers(this._nextValue, this.context);
        if( this._currentValue == null){
            this._currentValue = TransformableValues.fromInstance(this._nextValue);
            this._currentViewer.onNbElementsChanged(this._currentValue);
        }
        if (isReshaped){
            if (this._currentValue.nbChannels != this._nextValue.nbChannels){
                UserInterface.getInstance().nbChannels = this._nextValue.nbChannels;
                this._currentViewer.onNbChannelsChanged(this._currentValue)
            }
            if (this._currentValue.nbElements != this._nextValue.nbElements){
                this._currentValue = TransformableValues.fromInstance(this._nextValue);
                this._currentViewer.onNbElementsChanged(this._currentValue);
            }
        }
        if (!this._animationTimer.isRunning && this._needAnimationPlayOnReceived){
            this._needAnimationPlayOnReceived = false;
            this._needOneAnimationLoop = false;
            this.startVisualizationAnimation();
        }
        else if (!this._animationTimer.isRunning && this._needOneAnimationLoop){
            this._needOneAnimationLoop = false;
            this._currentViewer.updateScene(this._nextValue);
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
        this.updateScene();
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }

    public stopVisualizationAnimation() {
        this._currentViewer.updateScene(this._currentValue);
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
            this._currentViewer.updateScene(this._currentValue);
        }
        else{
            this._needOneAnimationLoop = true;
        }
        let values = TransformableValues.fromInstance(this._currentValue);
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.APPLY_INTERACTION,
                            [interaction, [mask].concat(values.toArray())], [mask.buffer].concat(values.toArrayBuffers()));
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

class TexturesContainer{
    private _posXTexture : WebGLTexture;
    private _posYTexture : WebGLTexture;
    private _statesTextures : Array<WebGLTexture>;

    public createBuffers(values : TransformableValues, context : WebGL2RenderingContext){
        let time = performance.now();
        let width = Math.ceil(Math.sqrt(values.nbElements));
        let height = width;
        this._posXTexture = context.createTexture();
        context.bindTexture(gl.TEXTURE_2D, this._posXTexture);
        context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT,
            values.states[0]);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this._posYTexture = context.createTexture();
        context.bindTexture(gl.TEXTURE_2D, this._posYTexture);
        context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT,
            values.states[0]);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this._statesTextures = new Array<WebGLTexture>(values.nbChannels);
        for (let i = 0; i < values.nbChannels; ++i){
            this._statesTextures[i] = context.createTexture();
            context.bindTexture(gl.TEXTURE_2D, this._statesTextures[i]);
            context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT,
                values.states[i]);
            context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        }
    }
}