import * as shaderUtils from "./shaderUtils.js";
import { Vec3 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Viewer } from "./viewer.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
const srcVertexShader = "/static/shaders/multipleMeshes.vert";
const srcFragmentShader = "/static/shaders/multipleMeshes.frag";
export class ViewerMultipleMeshes extends Viewer {
    _multipleInstances;
    _shaderProgram;
    _timeBuffer;
    _currentMeshFile;
    constructor(canvas, context, manager) {
        super(canvas, context, manager);
        this._currentMeshFile = "/static/models/roundedCube1.obj";
        this._shaderProgram = new shaderUtils.ProgramWithTransformer(context);
    }
    async initialization() {
        this.initCamera();
        await this._shaderProgram.generateProgram(srcVertexShader, srcFragmentShader);
        this.context.useProgram(this._shaderProgram.program);
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK);
        this.context.enable(gl.DEPTH_TEST);
        this._timeBuffer = this.context.createBuffer();
        let blockIdx = this.context.getUniformBlockIndex(this._shaderProgram.program, shaderUtils.ShaderBlockIndex.TIME);
        let bindingPoint = shaderUtils.ShaderBlockBindingPoint.TIME;
        this.context.bindBuffer(gl.UNIFORM_BUFFER, this._timeBuffer);
        this.context.bufferData(gl.UNIFORM_BUFFER, 4 * Object.values(shaderUtils.AnimableValue).length / 2, gl.DYNAMIC_DRAW);
        this.context.bindBufferBase(gl.UNIFORM_BUFFER, bindingPoint, this._timeBuffer);
        this.context.uniformBlockBinding(this._shaderProgram.program, blockIdx, bindingPoint);
    }
    ;
    initCamera() {
        const cameraPos = Vec3.fromValues(0., 80., 100.);
        const cameraTarget = Vec3.fromValues(0, 0, 0);
        const up = Vec3.fromValues(0, 1, 0);
        const fovy = Math.PI / 4;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        const near = 0.1;
        const far = 100000;
        this._camera = Camera.getPerspectiveCamera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
    }
    async initMesh(values) {
        this.isDrawable = false;
        if (this._multipleInstances != null)
            delete this._multipleInstances;
        this._multipleInstances = new MultipleMeshInstances(this.context, values);
        await this._multipleInstances.loadMesh(this._currentMeshFile);
        this.isDrawable = true;
    }
    onCanvasResize() {
        this._camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
    }
    updateScene(values) {
        return;
    }
    clear() {
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }
    draw(textures) {
        this.context.useProgram(this._shaderProgram.program);
        let projLoc = this.context.getUniformLocation(this._shaderProgram.program, "u_proj");
        let viewLoc = this.context.getUniformLocation(this._shaderProgram.program, "u_view");
        let lightLoc = this.context.getUniformLocation(this._shaderProgram.program, "u_light_loc");
        let dimensionLoc = this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderUniforms.DIMENSION);
        let aabb = this.context.getUniformLocation(this._shaderProgram.program, "u_aabb");
        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this._camera.viewMatrix);
        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform3fv(lightLoc, lightPos);
        this.context.uniform2f(dimensionLoc, textures.width, textures.height);
        // times
        let times = new Float32Array(Object.values(shaderUtils.AnimableValue).length / 2);
        for (let i = 0; i < Object.values(shaderUtils.AnimableValue).length / 2; i++) {
            times[i] = this._manager.getAnimationTime(i);
        }
        this.context.bindBuffer(gl.UNIFORM_BUFFER, this._timeBuffer);
        this.context.bufferSubData(gl.UNIFORM_BUFFER, 0, times);
        // ...
        if (textures.getTextures(0) != null) {
            let id = 0;
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D_ARRAY, textures.getTextures(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderElementInputs.TEX_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D_ARRAY, textures.getTextures(1));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderElementInputs.TEX_T1), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getMask());
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderElementInputs.TEX_SELECTION), id++);
        }
        this.context.uniform2fv(aabb, this._multipleInstances.aabb, 0, 0);
        this._multipleInstances.draw();
    }
    currentSelectionChanged(selection) {
        this._multipleInstances.updateMouseOverBuffer(selection);
    }
    onReset(newValues) {
        return;
    }
    async onNbElementsChanged(newValues) {
        await this.initMesh(newValues);
    }
    onNbChannelsChanged(newValues) {
        return;
    }
    updateProgamsTransformers(transformers) {
        this._shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock());
        if (this._shaderProgram.program) {
            const blockIdx = this.context.getUniformBlockIndex(this._shaderProgram.program, shaderUtils.ShaderBlockIndex.TIME);
            const bindingPoint = shaderUtils.ShaderBlockBindingPoint.TIME;
            this.context.uniformBlockBinding(this._shaderProgram.program, blockIdx, bindingPoint);
        }
    }
    onMouseMoved(deltaX, deltaY) {
        this._camera.rotateCamera(deltaX, deltaY);
    }
    onWheelMoved(delta) {
        this._camera.moveForward(-delta);
    }
    getViewBoundaries() {
        let xFactor = this._manager.transformers.getPositionFactor(0);
        let zFactor = this._manager.transformers.getPositionFactor(2);
        let textures = this._manager.currentTextures;
        let minX = textures.getXMin(0) * xFactor;
        let maxX = textures.getXMax(0) * xFactor;
        let minY = textures.getYMin(0) * zFactor;
        let maxY = textures.getYMax(0) * zFactor;
        return [minX, maxX, minY, maxY];
    }
}
