import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { UserInterface } from "./interface/userInterface.js";
import { ViewerMultipleMeshes } from "./viewerMultipleMeshes.js";
import { ViewerTexture } from "./viewerTexture.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export var ViewerType;
(function (ViewerType) {
    ViewerType["MULTIPLE_MESHES"] = "Meshes";
    ViewerType["TEXTURE"] = "Texture";
})(ViewerType || (ViewerType = {}));
export class ViewerManager {
    context;
    canvas;
    resizeObserver;
    transformers;
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
        this._viewers = [this._currentViewer, new ViewerTexture(this.canvas, this.context, this)];
        this._textures = new TexturesContainer(this.context);
    }
    set stats(stats) {
        this._stats = stats;
    }
    get currentTextures() {
        return this._textures;
    }
    get camera() {
        return this._currentViewer.camera;
    }
    async initialization(viewer) {
        this._viewers.forEach(viewer => {
            viewer.initialization();
        });
        this.switchViewer(viewer);
        let self = this;
        this.resizeObserver = new ResizeObserver(function () { self.onCanvasResize(); });
        this.resizeObserver.observe(this.canvas);
        this._animationTimer.callback = function () {
            this.updateScene();
        }.bind(this);
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.RESET);
    }
    switchViewer(viewer) {
        switch (viewer) {
            case ViewerType.MULTIPLE_MESHES:
                this._currentViewer = this._viewers[0];
                break;
            case ViewerType.TEXTURE:
                this._currentViewer = this._viewers[1];
                break;
        }
        this._currentViewer.onReset(this._values);
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
        this._viewers.forEach(e => { e.onCanvasResize(); });
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
        this._viewers.forEach(e => { e.onReset(this._values); });
    }
    onValuesReceived(data, isReshaped = false) {
        this._values = TransformableValues.fromValuesAsArray(data);
        if (isReshaped) {
            let isChannels = this._values.nbChannels != this._textures.nbChannels;
            let isElements = this._values.nbElements != this._textures.nbElements;
            if (isChannels) {
                UserInterface.getInstance().nbChannels = this._values.nbChannels;
                this._textures.createBuffers(this._values);
                this._viewers.forEach(e => e.onNbChannelsChanged(this._values));
            }
            if (isElements) {
                this._textures.updateBuffers(this._values);
                this._viewers.forEach(e => e.onNbElementsChanged(this._values));
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
        this.transformers = transformers;
        this._viewers.forEach(e => { e.updateProgamsTransformers(transformers); });
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
        sendMessageToWorker(this.transmissionWorker, WorkerMessage.APPLY_INTERACTION, [interaction, this._textures.currentId, mask], [mask.buffer]);
    }
    currentSelectionChanged(selection) {
        this._textures.updateMask(selection);
    }
    onMouseMoved(deltaX, deltaY) {
        this._currentViewer.onMouseMoved(deltaX, deltaY);
    }
    onWheelMoved(delta) {
        this._currentViewer.onWheelMoved(delta);
    }
    getViewBoundaries() {
        return this._currentViewer.getViewBoundaries();
    }
    createMaskTexture(width, height) {
        this._textures.createMask(width, height);
    }
    updateMaskTexture(mask) {
        this._textures.updateMask(mask);
    }
}
export class TexturesContainer {
    _maskTexture;
    _posXTexture;
    _posYTexture;
    _statesTextures;
    context;
    _nbSteps = 3;
    _step = 0;
    // used to bufferize next steps
    _currentT;
    _nbElements;
    _nbChannels;
    _currentId;
    _minX;
    _maxX;
    _minY;
    _maxY;
    _width;
    _height;
    _maskWidth;
    _maskHeight;
    constructor(context) {
        this.context = context;
        this._posXTexture = new Array(this._nbSteps);
        this._posYTexture = new Array(this._nbSteps);
        this._statesTextures = new Array(this._nbSteps);
        this._currentId = [0, 0, 0];
        this._minX = [0, 0, 0];
        this._maxX = [0, 0, 0];
        this._minY = [0, 0, 0];
        this._maxY = [0, 0, 0];
    }
    get nbElements() {
        return this._nbElements;
    }
    get nbChannels() {
        return this._nbChannels;
    }
    get currentId() {
        return this._currentId[this._step];
    }
    get width() {
        return this._width;
    }
    get height() {
        return this._height;
    }
    getStepWithT(t) {
        switch (this._step) {
            case 0:
                return t == 0 ? 0 : 1;
            case 1:
                return t == 0 ? 1 : 2;
            case 2:
                return t == 0 ? 2 : 0;
        }
    }
    getPosXTexture(t) {
        return this._posXTexture[this.getStepWithT(t)];
    }
    getPosYTexture(t) {
        return this._posYTexture[this.getStepWithT(t)];
    }
    getStatesTexture(t) {
        return this._statesTextures[this.getStepWithT(t)];
    }
    getXMin(t) {
        return this._minX[this.getStepWithT(t)];
    }
    getXMax(t) {
        return this._maxX[this.getStepWithT(t)];
    }
    getYMin(t) {
        return this._minY[this.getStepWithT(t)];
    }
    getYMax(t) {
        return this._maxY[this.getStepWithT(t)];
    }
    getMask() {
        return this._maskTexture;
    }
    createMask(width, height) {
        this._maskWidth = width;
        this._maskHeight = height;
        this._maskTexture = this.context.createTexture();
        this.context.bindTexture(gl.TEXTURE_2D, this._maskTexture);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        this.context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        let blank = new Float32Array(width * height).fill(-1);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this._maskWidth, this._maskHeight, 0, gl.RED, gl.FLOAT, blank);
    }
    updateMask(mask) {
        this.context.bindTexture(gl.TEXTURE_2D, this._maskTexture);
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this._maskWidth, this._maskHeight, 0, gl.RED, gl.FLOAT, mask);
    }
    createBuffers(values) {
        this._nbElements = values.nbElements;
        this._nbChannels = values.nbChannels;
        this._width = Math.ceil(Math.sqrt(values.nbElements));
        this._height = this._width;
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
        let width = Math.ceil(Math.sqrt(values.nbElements));
        let height = Math.floor(Math.sqrt(values.nbElements));
        let currentStep = (this._step + this._currentT) % this._nbSteps;
        this._currentId[currentStep] = values.id;
        let boundsX = values.getBoundsX();
        let boundsY = values.getBoundsY();
        this._minX[currentStep] = boundsX[0];
        this._maxX[currentStep] = boundsX[1];
        this._minY[currentStep] = boundsY[0];
        this._maxY[currentStep] = boundsY[1];
        let fillEnd = false;
        if (width * height > this.nbElements)
            fillEnd = true;
        this.context.bindTexture(gl.TEXTURE_2D, this._posXTexture[currentStep]);
        let source = values.positionX;
        if (fillEnd) {
            source = new Float32Array(width * height).fill(0., this.nbElements);
            source.set(values.positionX, 0);
        }
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);
        source = values.positionY;
        this.context.bindTexture(gl.TEXTURE_2D, this._posYTexture[currentStep]);
        if (fillEnd) {
            source = new Float32Array(width * height).fill(0., this.nbElements);
            source.set(values.positionY, 0);
        }
        this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);
        for (let i = 0; i < values.nbChannels; ++i) {
            this.context.bindTexture(gl.TEXTURE_2D, this._statesTextures[currentStep][i]);
            source = values.states[i];
            if (fillEnd) {
                source = new Float32Array(width * height).fill(0., this.nbElements);
                source.set(values.states[i], 0);
            }
            this.context.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, source);
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
