import * as shaderUtils from "./shaderUtils.js";
import { Vec3 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Stats } from "./stats.js";
import { AnimationTimer } from "./animationTimer.js";
import { StatesBuffer } from "./statesBuffer.js";
import { StatesTransformer } from "./statesTransformer.js";
import { SelectionHandler } from "./selectionHandler.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export var AnimableValue;
(function (AnimableValue) {
    AnimableValue[AnimableValue["COLOR"] = 0] = "COLOR";
    AnimableValue[AnimableValue["TRANSLATION"] = 1] = "TRANSLATION";
})(AnimableValue || (AnimableValue = {}));
export class Viewer {
    context;
    canvas;
    shaderProgram;
    resizeObserver;
    camera;
    _multipleInstances;
    _selectionHandler;
    _lastTime = 0;
    _stats;
    _animationTimer;
    _animationIds;
    _statesBuffer;
    _drawable;
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        let context = this.canvas.getContext("webgl2");
        if (context == null) {
            throw "Could not create WebGL2 context";
        }
        this.context = context;
        this._stats = new Stats(document.getElementById("renderingFps"), document.getElementById("updateMs"), document.getElementById("renderingMs"));
        this._animationTimer = new AnimationTimer(0.15, false);
        this._animationIds = [null, null];
        this._selectionHandler = SelectionHandler.getInstance(context);
        this._statesBuffer = new StatesBuffer(new StatesTransformer());
        this._drawable = false;
        this.shaderProgram = new shaderUtils.ProgramWithTransformer(context);
    }
    // initialization methods
    async initialization(srcVs, srcFs, nbInstances) {
        await this.shaderProgram.generateProgram(srcVs, srcFs);
        this.context.useProgram(this.shaderProgram.program);
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK);
        this.context.enable(gl.DEPTH_TEST);
        await this._selectionHandler.initialization("/static/shaders/selection.vert", "/static/shaders/selection.frag");
        this.initCamera();
        let self = this;
        this.resizeObserver = new ResizeObserver(function () { self.onCanvasResize(); });
        this.resizeObserver.observe(this.canvas);
        this._animationTimer.callback = function () {
            this.updateScene();
        }.bind(this);
        await this.initCurrentVisu(nbInstances);
        this._drawable = true;
    }
    async initCurrentVisu(nbElements) {
        this._drawable = false;
        this._statesBuffer.initializeElements(nbElements);
        while (!this._statesBuffer.isReady) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
        let values = this._statesBuffer.values;
        await this.initMesh(values);
        this._drawable = true;
    }
    async initMesh(values) {
        if (this._multipleInstances != null)
            delete this._multipleInstances;
        this._multipleInstances = new MultipleMeshInstances(this.context, values);
        await this._multipleInstances.loadMesh("/static/models/cube_div_1.obj");
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
    // private methods
    onCanvasResize() {
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this._selectionHandler.resizeBuffers();
    }
    updateScene() {
        let values = this._statesBuffer.values;
        this._multipleInstances.updateStates(values);
    }
    clear() {
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }
    draw() {
        this.context.useProgram(this.shaderProgram.program);
        let projLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_proj");
        let viewLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_view");
        let lightLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_light_loc");
        let timeColorLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_time_color");
        let timeTranslationLoc = this.context.getUniformLocation(this.shaderProgram.program, "u_time_translation");
        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this.camera.viewMatrix);
        this.context.uniformMatrix4fv(projLoc, false, this.camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this.camera.viewMatrix);
        this.context.uniform3fv(lightLoc, lightPos);
        this.context.uniform1f(timeColorLoc, this.getAnimationTime(AnimableValue.COLOR));
        this.context.uniform1f(timeTranslationLoc, this.getAnimationTime(AnimableValue.TRANSLATION));
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
        this._stats.startRenderingTimer(delta);
        this.clear();
        // let prevSelection = this._selectionHandler.selectedId;
        // this._selectionHandler.updateCurrentSelection(this.camera, this._multipleInstances, this.getAnimationTime(AnimableValue.TRANSLATION));
        // let currentSelection = this._selectionHandler.selectedId;
        // if (this._selectionHandler.hasCurrentSelection() && currentSelection != prevSelection){
        //     this._multipleInstances.setMouseOver(currentSelection);
        // }
        if (this._drawable)
            this.draw();
        this.context.finish();
        this._stats.stopRenderingTimer();
    }
    getAnimationTime(type) {
        let id = this._animationIds[type];
        if (id == null)
            return this._animationTimer.getAnimationTime();
        return this._animationTimer.getAnimationTime(id);
    }
    setCurrentTransformer(transformer) {
        this._statesBuffer.transformer = transformer;
    }
    bindAnimationCurve(type, fct) {
        let id = this._animationTimer.addAnimationCurve(fct);
        this._animationIds[type] = id;
    }
    startVisualizationAnimation() {
        this._animationTimer.loop = true;
        this._animationTimer.play();
    }
    stopVisualizationAnimation() {
        this._animationTimer.loop = false;
        this._animationTimer.stop();
    }
    get statesBuffer() {
        return this._statesBuffer;
    }
}
