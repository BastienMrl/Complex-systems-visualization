import { Camera } from "./camera.js";
import { Vec3 } from "./ext/glMatrix/vec3.js";
import { PlanMesh } from "./mesh.js";
import { ProgramWithTransformer, ShaderElementInputs, ShaderUniforms } from "./shaderUtils.js";
import { TransformFlag } from "./transformer/transformType.js";
import { Viewer } from "./viewer.js";
const gl = WebGL2RenderingContext;
const srcVertexShader = "/static/shaders/plan.vert";
const srcFragmentShader = "/static/shaders/plan.frag";
export class ViewerTexture extends Viewer {
    _shaderProgram;
    _mesh;
    static _meshSize = 10;
    constructor(canvas, context, manager) {
        super(canvas, context, manager);
        this._shaderProgram = new ProgramWithTransformer(context, false);
        this._mesh = new PlanMesh(context, ViewerTexture._meshSize);
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
        this.isDrawable = true;
        return;
    }
    initCamera() {
        const cameraPos = Vec3.fromValues(0., 10., 0.001);
        const cameraTarget = Vec3.fromValues(0, 0, 0);
        const up = Vec3.fromValues(0, 1, 0);
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        const near = 0.01;
        const far = 10000;
        this._camera = Camera.getOrthographicCamera(cameraPos, cameraTarget, up, aspect, near, far);
        this._camera.distanceMin = 0.5;
        this._camera.distanceMax = 50;
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
        let domainLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.POS_DOMAIN);
        let dimensionLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.DIMENSION);
        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform4f(domainLoc, textures.getXMin(0), textures.getXMax(0), textures.getYMin(0), textures.getYMax(0));
        this.context.uniform2f(dimensionLoc, textures.width, textures.height);
        if (textures.getStatesTexture(0) != null) {
            let id = 0;
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosXTexture(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_POS_X_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getPosYTexture(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_POS_Y_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getStatesTexture(0)[0]);
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_STATE_0_T0), id++);
            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getMask());
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_SELECTION), id++);
        }
        this._mesh.draw();
    }
    currentSelectionChanged(selection) {
        throw new Error("Method not implemented.");
    }
    onReset(newValues) {
        return;
    }
    onNbElementsChanged(newValues) {
        return;
    }
    onNbChannelsChanged(newValues) {
        return;
    }
    updateProgamsTransformers(transformers) {
        this._shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock(false, TransformFlag.COLOR));
    }
    onMouseMoved(deltaX, deltaY) {
        this._camera.move(-deltaY * this._camera.distance * 0.5, -deltaX * this._camera.distance * 0.5 * this._camera.aspectRatio);
    }
    onWheelMoved(delta) {
        this._camera.moveForward(-delta);
    }
    getViewBoundaries() {
        let length = ViewerTexture._meshSize / 2;
        return [-length, length, -length, length];
    }
}
