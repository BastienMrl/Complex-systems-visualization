import { ShaderUniforms, ProgramWithTransformer, ShaderElementInputs, initShaders } from "./shaderUtils.js";
import { Vec3, Mat4 } from "./ext/glMatrix/index.js";
import { Camera } from "./camera.js";
import { Mesh } from "./mesh.js";
import { Viewer } from "./viewer.js";
import { TransformFlag } from "./transformer/transformType.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
const srcVertexShader = "/static/shaders/material.vert";
const srcFragmentShader = "/static/shaders/material.frag";
const srcVertexPoint = "/static/shaders/point.vert";
const srcFragmentPoint = "/static/shaders/point.frag";
export class ViewerMaterial extends Viewer {
    _mesh;
    _shaderProgram;
    _shaderPoint;
    _currentMeshFile;
    constructor(canvas, context, manager) {
        super(canvas, context, manager);
        this._currentMeshFile = "/static/models/plan_bis.obj";
        this._mesh = new Mesh(context, this._currentMeshFile);
        this._shaderProgram = new ProgramWithTransformer(context, false);
    }
    async initialization() {
        this.initCamera();
        await this._shaderProgram.generateProgram(srcVertexShader, srcFragmentShader);
        this._shaderPoint = await initShaders(this.context, srcVertexPoint, srcFragmentPoint);
        this.context.useProgram(this._shaderProgram.program);
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK);
        this.context.enable(gl.DEPTH_TEST);
        this.isDrawable = true;
    }
    ;
    initCamera() {
        const cameraPos = Vec3.fromValues(0., 10., 10.);
        const cameraTarget = Vec3.fromValues(0, 0, 0);
        const up = Vec3.fromValues(0, 1, 0);
        const fovy = Math.PI / 4;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        const near = 0.1;
        const far = 100000;
        this._camera = Camera.getPerspectiveCamera(cameraPos, cameraTarget, up, fovy, aspect, near, far);
        this._camera.distanceMin = 0.5;
        this._camera.distanceMax = 100;
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
        let domainLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.POS_DOMAIN);
        let dimensionLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.DIMENSION);
        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform4f(domainLoc, textures.getXMin(0), textures.getXMax(0), textures.getYMin(0), textures.getYMax(0));
        this.context.uniform2f(dimensionLoc, textures.width, textures.height);
        let lightPos = Vec3.fromValues(0.0, 0.0, 0.0);
        let transform = Mat4.create();
        Mat4.fromRotation(transform, performance.now() * 0.001, Vec3.fromValues(0., 1., 0.));
        transform.rotate(Math.PI / 4, Vec3.fromValues(0., 0., 1.));
        transform.translate(Vec3.fromValues(0., 2., 0));
        Vec3.transformMat4(lightPos, lightPos, transform);
        let transformedLightPos = Vec3.create();
        Vec3.transformMat4(transformedLightPos, lightPos, this._camera.viewMatrix);
        this.context.uniform3fv(lightLoc, transformedLightPos);
        if (textures.getTextures(0) != null) {
            let id = 0;
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D_ARRAY, textures.getTextures(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getMask());
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_SELECTION), id++);
        }
        this._mesh.draw();
        this.context.useProgram(this._shaderPoint);
        projLoc = this.context.getUniformLocation(this._shaderPoint, "u_proj");
        viewLoc = this.context.getUniformLocation(this._shaderPoint, "u_view");
        let colorLoc = this.context.getUniformLocation(this._shaderPoint, "u_color");
        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform3f(colorLoc, 0.9, 0.9, 0.9);
        let posBuffer = this.context.createBuffer();
        let pos = new Float32Array(lightPos);
        this.context.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        this.context.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
        this.context.vertexAttribPointer(0, 3, gl.FLOAT, false, 0., 0.);
        this.context.enableVertexAttribArray(0);
        this.context.bindBuffer(gl.ARRAY_BUFFER, null);
        this.context.drawArrays(gl.POINTS, 0, 1);
    }
    currentSelectionChanged(selection) {
        return;
    }
    onReset(newValues) {
        return;
    }
    async onNbElementsChanged(newValues) {
        return;
    }
    onNbChannelsChanged(newValues) {
        return;
    }
    updateProgamsTransformers(transformers) {
        this._shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock(false, TransformFlag.COLOR));
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
