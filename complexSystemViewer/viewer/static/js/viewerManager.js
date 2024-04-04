import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { UserInterface } from "./interface/userInterface.js";
import { ViewerMultipleMeshes } from "./viewerMultipleMeshes.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export var ViewerType;
(function (ViewerType) {
    ViewerType[ViewerType["MULTIPLE_MESHES"] = 0] = "MULTIPLE_MESHES";
    ViewerType[ViewerType["TEXTURE"] = 1] = "TEXTURE";
})(ViewerType || (ViewerType = {}));
export class ViewerManager {
    context;
    canvas;
    resizeObserver;
    _viewers;
    _currentViewer;
    _lastTime = 0;
    _stats;
    transmissionWorker;
    _animationTimer;
    _animationIds;
    _needAnimationPlayOnReceived = false;
    _needOneAnimationLoop = false;
    _values;
    _textures;
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        let context = this.canvas.getContext("webgl2");
        if (context == null) {
            throw "Could not create WebGL2 context";
        }
        this.context = context;
        this._animationTimer = new AnimationTimer(0.15, false);
        this._animationIds = new Map();
        this._values = null;
        this.transmissionWorker = new Worker("/static/js/workers/transmissionWorker.js", { type: "module" });
        this.transmissionWorker.onmessage = this.onTransmissionWorkerMessage.bind(this);
        this._currentViewer = new ViewerMultipleMeshes(this.canvas, this.context, this);
        this._viewers = [this._currentViewer];
        this._textures = new TexturesContainer(this.context);
    }
    set stats(stats) {
        this._stats = stats;
    }
    async initialization(viewer) {
        switch (viewer) {
            case ViewerType.MULTIPLE_MESHES:
                this._currentViewer = this._viewers[0];
            case ViewerType.TEXTURE:
                break;
        }
        this._currentViewer.initialization();
        let self = this;
        this.resizeObserver = new ResizeObserver(function () { self.onCanvasResize(); });
        this.resizeObserver.observe(this.canvas);
        this._animationTimer.callback = function () {
            this.updateScene();
        }.bind(this);
        this._currentViewer.isDrawable = false;
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.RESET);
    }
    // in seconds
    set animationDuration(duration) {
        this._animationTimer.duration = duration;
    }
    // private methods
    onCanvasResize() {
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this._currentViewer.onCanvasResize();
    }
    updateScene() {
        if (this._values == null) {
            return;
        }
        this._stats.startUpdateTimer();
        this._values = null;
        this._textures.step();
        this.context.finish();
        this._stats.stopUpdateTimer();
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.GET_VALUES);
    }
    loopAnimation() {
        const loop = (time) => {
            this.render(time);
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }
    // public methods
    render(time) {
        time *= 0.001;
        let delta = this._lastTime = 0 ? 0 : time - this._lastTime;
        this._lastTime = time;
        // rendering
        if (this._currentViewer.isDrawable) {
            this._stats.startRenderingTimer(delta);
            this._currentViewer.clear();
            this._currentViewer.draw(this._textures);
            this.context.finish();
            this._stats.stopRenderingTimer();
        }
    }
    getAnimationTime(type) {
        let id = this._animationIds.get(type);
        if (id == undefined || id == null)
            return this._animationTimer.getAnimationTime();
        return this._animationTimer.getAnimationTime(id);
    }
    onTransmissionWorkerMessage(e) {
        switch (getMessageHeader(e)) {
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
                this._stats.displayWorkerTimer(getMessageBody(e)[0], getMessageBody(e)[1]);
                break;
        }
    }
    async onReset() {
        this._values = null;
        while (this._values == null) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
        this._textures.createBuffers(this._values);
        this._currentViewer.onReset(this._values);
    }
    async onValuesReceived(data, isReshaped = false) {
        this._values = TransformableValues.fromValuesAsArray(data);
        if (isReshaped) {
            let isChannels = this._values.nbChannels != this._textures.nbChannels;
            let isElements = this._values.nbElements != this._textures.nbElements;
            if (isChannels) {
                UserInterface.getInstance().nbChannels = this._values.nbChannels;
                this._textures.createBuffers(this._values);
                this._currentViewer.onNbChannelsChanged(this._values);
            }
            if (isElements) {
                this._textures.updateBuffers(this._values);
                this._currentViewer.onNbElementsChanged(this._values);
            }
        }
        else {
            this._textures.updateBuffers(this._values);
        }
        if (!this._animationTimer.isRunning && this._needAnimationPlayOnReceived) {
            this._needAnimationPlayOnReceived = false;
            this._needOneAnimationLoop = false;
            this.startVisualizationAnimation();
        }
        else if (!this._animationTimer.isRunning && this._needOneAnimationLoop) {
            this._needOneAnimationLoop = false;
            this._textures.updateBuffers(this._values);
            this.startOneAnimationLoop();
        }
    }
    bindAnimationCurve(type, fct) {
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds.set(type, id);
    }
    startVisualizationAnimation() {
        if (this._animationTimer.isRunning) {
            if (!this._animationTimer.loop)
                this._animationTimer.loop = true;
            return;
        }
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }
    stopVisualizationAnimation() {
        this._animationTimer.stop();
    }
    startOneAnimationLoop() {
        if (this._animationTimer.isRunning)
            return;
        this._animationTimer.loop = false;
        // TODO set duration
        this._animationTimer.play();
    }
    updateProgamsTransformers(transformers) {
        this._currentViewer.updateProgamsTransformers(transformers);
    }
    sendInteractionRequest(mask, interaction = "0") {
        if (this._animationTimer.isRunning && this._animationTimer.loop) {
            this.stopVisualizationAnimation();
            this._needAnimationPlayOnReceived = true;
            this._textures.updateBuffers(this._values);
        }
        else {
            this._needOneAnimationLoop = true;
        }
        let values = TransformableValues.fromInstance(this._values);
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.APPLY_INTERACTION, [interaction, [mask].concat(values.toArray())], [mask.buffer].concat(values.toArrayBuffers()));
    }
    getElementOver(posX, posY) {
        return this._currentViewer.getElementOver(posX, posY);
    }
    currentSelectionChanged(selection) {
        this._currentViewer.currentSelectionChanged(selection);
    }
    onMouseMoved(deltaX, deltaY) {
        this._currentViewer.onMouseMoved(deltaX, deltaY);
    }
    onWheelMoved(delta) {
        this._currentViewer.onWheelMoved(delta);
    }
}
export class TexturesContainer {
    _posXTexture;
    _posYTexture;
    _statesTextures;
    context;
    _step = 0;
    _currentT;
    _nbSteps = 3;
    _nbElements;
    _nbChannels;
    constructor(context) {
        this.context = context;
        this._posXTexture = new Array(this._nbSteps);
        this._posYTexture = new Array(this._nbSteps);
        this._statesTextures = new Array(this._nbSteps);
    }
    get nbElements() {
        return this._nbElements;
    }
    get nbChannels() {
        return this._nbChannels;
    }
    getPosXTexture(t) {
        switch (this._step) {
            case 0:
                return t == 0 ? this._posXTexture[0] : this._posXTexture[1];
            case 1:
                return t == 0 ? this._posXTexture[1] : this._posXTexture[2];
            case 2:
                return t == 0 ? this._posXTexture[2] : this._posXTexture[0];
        }
    }
    getPosYTexture(t) {
        switch (this._step) {
            case 0:
                return t == 0 ? this._posYTexture[0] : this._posYTexture[1];
            case 1:
                return t == 0 ? this._posYTexture[1] : this._posYTexture[2];
            case 2:
                return t == 0 ? this._posYTexture[2] : this._posYTexture[0];
        }
    }
    getStatesTexture(t) {
        switch (this._step) {
            case 0:
                return t == 0 ? this._statesTextures[0] : this._statesTextures[1];
            case 1:
                return t == 0 ? this._statesTextures[1] : this._statesTextures[2];
            case 2:
                return t == 0 ? this._statesTextures[2] : this._statesTextures[0];
        }
    }
    createBuffers(values) {
        this._nbElements = values.nbElements;
        this._nbChannels = values.nbChannels;
        for (let i = 0; i < this._nbSteps; ++i) {
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
            this._statesTextures[i] = new Array(values.nbChannels);
            for (let k = 0; k < values.nbChannels; ++k) {
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
    updateBuffers(values) {
        this._nbElements = values.nbElements;
        let t = this._currentT;
        let width = Math.ceil(Math.sqrt(values.nbElements));
        let height = width;
        this.context.bindTexture(gl.TEXTURE_2D, this._posXTexture[(this._step + t) % this._nbSteps]);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values.positionX);
        this.context.bindTexture(gl.TEXTURE_2D, this._posYTexture[(this._step + t) % this._nbSteps]);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values.positionY);
        for (let i = 0; i < values.nbChannels; ++i) {
            this.context.bindTexture(gl.TEXTURE_2D, this._statesTextures[(this._step + t) % this._nbSteps][i]);
            this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values.states[i]);
        }
        this._currentT += 1;
        if (this._currentT >= this._nbSteps)
            this._currentT = 2;
    }
    step() {
        switch (this._step) {
            case 0:
                this._step = 1;
                break;
            case 1:
                this._step = 2;
                break;
            case 2:
                this._step = 0;
                break;
        }
    }
}
