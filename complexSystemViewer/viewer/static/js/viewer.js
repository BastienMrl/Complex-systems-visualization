import * as shaderUtils from "./shaderUtils.js";
import { Vec3 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { AnimationTimer } from "./animationTimer.js";
import { TransformableValues } from "./transformableValues.js";
import { WorkerMessage, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workers/workerInterface.js";
import { SelectionManager } from "./interface/selectionTools/selectionManager.js";
import { UserInterface } from "./interface/userInterface.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export class Viewer {
    context;
    canvas;
    shaderProgram;
    resizeObserver;
    camera;
    _multipleInstances;
    _selectionManager;
    _lastTime = 0;
    _stats;
    _animationTimer;
    _animationIds;
    _needAnimationPlayOnReceived = false;
    _needOneAnimationLoop = false;
    _transmissionWorker;
    _currentValue;
    _nextValue;
    _drawable;
    _currentMeshFile;
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        let context = this.canvas.getContext("webgl2");
        if (context == null) {
            throw "Could not create WebGL2 context";
        }
        this.context = context;
        this._animationTimer = new AnimationTimer(0.15, false);
        this._animationIds = new Map();
        this._selectionManager = new SelectionManager(this);
        this._currentValue = null;
        this._nextValue = null;
        this._transmissionWorker = new Worker("/static/js/workers/transmissionWorker.js", { type: "module" });
        this._transmissionWorker.onmessage = this.onTransmissionWorkerMessage.bind(this);
        this._drawable = false;
        this._currentMeshFile = "/static/models/roundedCube1.obj";
        this.shaderProgram = new shaderUtils.ProgramWithTransformer(context);
    }
    set stats(stats) {
        this._stats = stats;
        this._selectionManager.stats = stats;
    }
    // initialization methods
    async initialization(srcVs, srcFs) {
        await this.shaderProgram.generateProgram(srcVs, srcFs);
        this.context.useProgram(this.shaderProgram.program);
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK);
        this.context.enable(gl.DEPTH_TEST);
        this.initCamera();
        let self = this;
        this.resizeObserver = new ResizeObserver(function () { self.onCanvasResize(); });
        this.resizeObserver.observe(this.canvas);
        this._animationTimer.callback = function () {
            this.updateScene();
        }.bind(this);
        await this.initCurrentVisu();
    }
    async initCurrentVisu() {
        this._drawable = false;
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.RESET);
    }
    async initMesh(values) {
        this._drawable = false;
        if (this._multipleInstances != null)
            delete this._multipleInstances;
        this._multipleInstances = new MultipleMeshInstances(this.context, values);
        this._selectionManager.setMeshes(this._multipleInstances);
        await this._multipleInstances.loadMesh(this._currentMeshFile);
        this._drawable = true;
    }
    loadMesh(path) {
        this._multipleInstances.loadMesh(path);
    }
    initCamera() {
        const cameraPos = Vec3.fromValues(0., 80., 100.);
        const cameraTarget = Vec3.fromValues(0, 0, 0);
        const up = Vec3.fromValues(0, 1, 0);
        const fovy = Math.PI / 4;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        const near = 0.1;
        const far = 100000;
        this.camera = new Camera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
    }
    get selectionManager() {
        return this._selectionManager;
    }
    get transmissionWorker() {
        return this._transmissionWorker;
    }
    // setter
    set currentMeshFile(meshFile) {
        this._currentMeshFile = meshFile;
    }
    // in seconds
    set animationDuration(duration) {
        this._animationTimer.duration = duration;
    }
    // private methods
    onCanvasResize() {
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    updateScene() {
        if (this._nextValue == null) {
            return;
        }
        this._stats.startUpdateTimer();
        this._currentValue = this._nextValue;
        this._multipleInstances.updateStates(this._nextValue);
        this._nextValue = null;
        this.context.finish();
        this._stats.stopUpdateTimer();
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.GET_VALUES);
    }
    clear() {
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }
    draw() {
        this.context.useProgram(this.shaderProgram.program);
        let projLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_proj");
        let viewLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_view");
        let lightLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_light_loc");
        let aabb = this.context.getUniformLocation(this.shaderProgram.program, "u_aabb");
        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this.camera.viewMatrix);
        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);
        this.context.uniform3fv(lightLoc, lightPos);
        for (let i = 0; i < Object.values(shaderUtils.AnimableValue).length / 2; i++) {
            let location = this.context.getUniformLocation(this.shaderProgram.program, shaderUtils.getAnimableValueUniformName(i));
            this.context.uniform1f(location, this.getAnimationTime(i));
        }
        this.context.uniform2fv(aabb, this._multipleInstances.aabb, 0, 0);
        this._multipleInstances.draw();
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
        if (this._drawable) {
            this._stats.startRenderingTimer(delta);
            this.clear();
            this.draw();
            this.context.finish();
            this._stats.stopRenderingTimer();
        }
    }
    currentSelectionChanged(selection) {
        this._multipleInstances.updateMouseOverBuffer(selection);
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
                this._stats.logShape(this._currentValue.nbElements, this._currentValue.nbChannels);
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
        if (this._multipleInstances == null)
            return;
        this._nextValue = null;
        while (this._nextValue == null) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
        this._multipleInstances.updateStates(this._nextValue);
        this._multipleInstances.updateStates(this._nextValue);
        this._currentValue = TransformableValues.fromInstance(this._nextValue);
    }
    async onValuesReceived(data, isReshaped = false) {
        this._nextValue = TransformableValues.fromValuesAsArray(data);
        if (this._currentValue == null) {
            this._currentValue = TransformableValues.fromInstance(this._nextValue);
            await this.initMesh(this._nextValue);
        }
        if (isReshaped) {
            if (this._currentValue.nbChannels != this._nextValue.nbChannels) {
                UserInterface.getInstance().nbChannels = this._nextValue.nbChannels;
            }
            if (this._currentValue.nbElements != this._nextValue.nbElements) {
                this._currentValue = TransformableValues.fromInstance(this._nextValue);
                await this.initMesh(this._nextValue);
            }
        }
        if (!this._animationTimer.isRunning && this._needAnimationPlayOnReceived) {
            this._needAnimationPlayOnReceived = false;
            this._needOneAnimationLoop = false;
            this.startVisualizationAnimation();
        }
        else if (!this._animationTimer.isRunning && this._needOneAnimationLoop) {
            this._needOneAnimationLoop = false;
            this._multipleInstances.updateStates(this._nextValue);
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
        this.updateScene();
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }
    stopVisualizationAnimation() {
        this._multipleInstances.updateStates(this._currentValue);
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
        this.shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock());
    }
    sendInteractionRequest(mask, interaction = "0") {
        if (this._animationTimer.isRunning && this._animationTimer.loop) {
            this.stopVisualizationAnimation();
            this._needAnimationPlayOnReceived = true;
            this._multipleInstances.updateStates(this._currentValue);
        }
        else {
            this._needOneAnimationLoop = true;
        }
        let values = TransformableValues.fromInstance(this._currentValue);
        sendMessageToWorker(this._transmissionWorker, WorkerMessage.APPLY_INTERACTION, [interaction, [mask].concat(values.toArray())], [mask.buffer].concat(values.toArrayBuffers()));
    }
}
