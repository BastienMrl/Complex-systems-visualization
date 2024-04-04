import * as shaderUtils from "./shaderUtils.js";
import { Vec3 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { MultipleMeshInstances } from "./mesh.js";
import { Viewer } from "./viewer.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
const srcVertexShader = "/static/shaders/simple.vert";
const srcFragmentShader = "/static/shaders/simple.frag";
export class ViewerMultipleMeshes extends Viewer {
    _camera;
    _multipleInstances;
    _shaderProgram;
    _transformers;
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
        this._camera = new Camera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
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
        let aabb = this.context.getUniformLocation(this._shaderProgram.program, "u_aabb");
        let lightPos = Vec3.fromValues(0.0, 100.0, 10.0);
        Vec3.transformMat4(lightPos, lightPos, this._camera.viewMatrix);
        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform3fv(lightLoc, lightPos);
        // times
        let times = new Float32Array(Object.values(shaderUtils.AnimableValue).length / 2);
        for (let i = 0; i < Object.values(shaderUtils.AnimableValue).length / 2; i++) {
            times[i] = this._manager.getAnimationTime(i);
        }
        this.context.bindBuffer(gl.UNIFORM_BUFFER, this._timeBuffer);
        this.context.bufferSubData(gl.UNIFORM_BUFFER, 0, times);
        // ...
        if (textures.getStatesTexture(0) != null) {
            let id = 0;
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosXTexture(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_POS_X_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosYTexture(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_POS_Y_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getStatesTexture(0)[0]);
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_STATE_0_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosXTexture(1));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_POS_X_T1), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosYTexture(1));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_POS_Y_T1), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getStatesTexture(1)[0]);
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, shaderUtils.ShaderMeshInputs.TEX_STATE_0_T1), id++);
        }
        this.context.uniform2fv(aabb, this._multipleInstances.aabb, 0, 0);
        this._multipleInstances.draw();
    }
    getElementOver(posX, posY) {
        let origin = this._camera.position;
        let x = (2.0 * posX) / this.canvas.width - 1.0;
        let y = 1.0 - (2.0 * posY) / this.canvas.height;
        let z = 1.0;
        let direction = Vec3.fromValues(x, y, z);
        Vec3.transformMat4(direction, direction, this._camera.projViewMatrix.invert());
        direction.normalize();
        let normal = Vec3.fromValues(0, 1, 0);
        let pNear = Vec3.fromValues(0, this._multipleInstances.localAabb[3], 0);
        let pFar = Vec3.fromValues(0, this._multipleInstances.localAabb[2], 0);
        let denominator = Vec3.dot(normal, direction);
        let tFar = -1;
        let tNear = -1;
        if (denominator != 0) {
            let p = Vec3.create();
            Vec3.sub(p, pFar, origin);
            tFar = Vec3.dot(p, normal) / denominator;
            Vec3.sub(p, pNear, origin);
            tNear = Vec3.dot(p, normal) / denominator;
        }
        if (tFar < 0 || tNear < 0)
            return null;
        const nbSample = 4;
        const tDelta = tFar - tNear;
        for (let i = 0; i < nbSample; i++) {
            let step = (i) / (nbSample - 1);
            let t = tNear + tDelta * step;
            let position = Vec3.create();
            let dir = Vec3.create();
            Vec3.copy(dir, direction);
            dir.scale(t);
            Vec3.add(position, origin, dir);
            let id = this.getMeshIdFromPos(position[0], position[2]);
            if (id != null)
                return id;
        }
        return null;
    }
    getMeshIdFromPos(x, z) {
        let offsetX = (this._multipleInstances.nbCol - 1);
        let offsetZ = (this._multipleInstances.nbRow - 1);
        let aabb = this._multipleInstances.localAabb;
        let xMin = (x + aabb[0]) / this._transformers.getPositionFactor(0);
        let xMax = (x + aabb[1]) / this._transformers.getPositionFactor(0);
        let zMin = (z + aabb[4]) / this._transformers.getPositionFactor(2);
        let zMax = (z + aabb[5]) / this._transformers.getPositionFactor(2);
        xMin += offsetX / 2;
        xMax += offsetX / 2;
        zMin += offsetZ / 2;
        zMax += offsetZ / 2;
        if (xMin == xMax)
            x = Math.round(xMax);
        else if (Number.isInteger(xMin) && Number.isInteger(xMax))
            x = xMin;
        else if (Math.ceil(xMin) == Math.floor(xMax))
            x = Math.ceil(xMin);
        if (zMin == zMax)
            z = Math.round(zMax);
        else if (Number.isInteger(zMin) && Number.isInteger(zMax))
            z = zMax;
        else if (Math.ceil(zMin) == Math.floor(zMax))
            z = Math.ceil(zMin);
        if (z >= this._multipleInstances.nbRow || z < 0 || x >= this._multipleInstances.nbCol || x < 0)
            return null;
        return this.coordToId(z, x);
    }
    coordToId(i, j) {
        return i * this._multipleInstances.nbCol + j;
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
        this._transformers = transformers;
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
}
