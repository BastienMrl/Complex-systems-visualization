import { Camera } from "./camera.js";
import { Vec3 } from "./ext/glMatrix/vec3.js";
import { PlanMesh } from "./mesh.js";
import { ProgramWithTransformer, ShaderElementInputs, ShaderUniforms } from "./shaderUtils.js";
import { TransformableValues } from "./transformableValues.js";
import { TransformFlag } from "./transformer/transformType.js";
import { TransformerBuilder } from "./transformer/transformerBuilder.js";
import { Viewer } from "./viewer.js";
import { TexturesContainer, ViewerManager } from "./viewerManager.js";

const gl = WebGL2RenderingContext
const srcVertexShader = "/static/shaders/plan.vert";
const srcFragmentShader = "/static/shaders/plan.frag";



export class ViewerTexture extends Viewer{
    private _shaderProgram : ProgramWithTransformer;
    private _mesh : PlanMesh;

    private static readonly _meshSize = 10


    public constructor(canvas : HTMLCanvasElement, context : WebGL2RenderingContext, manager : ViewerManager){
        super(canvas, context, manager);
        this._shaderProgram = new ProgramWithTransformer(context, false);
        this._mesh = new PlanMesh(context, ViewerTexture._meshSize);
    }

    public async initialization() {
        this.initCamera();


        await this._shaderProgram.generateProgram(srcVertexShader, srcFragmentShader);

        this.context.useProgram(this._shaderProgram.program);
        this.context.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.context.clearColor(0.2, 0.2, 0.2, 1.0);
        this.context.enable(gl.CULL_FACE);
        this.context.cullFace(gl.BACK)
        this.context.enable(gl.DEPTH_TEST);
        this.isDrawable = true;
        return
    }

    protected initCamera(){
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

    public onCanvasResize() {
        this._camera.aspectRatio = this.canvas.clientWidth / this.canvas.clientHeight;
    }
    public updateScene(values: TransformableValues) {
        return;
    }
    public clear() {
        this.context.clear(this.context.COLOR_BUFFER_BIT | this.context.DEPTH_BUFFER_BIT);
    }
    public draw(textures : TexturesContainer) {
        this.context.useProgram(this._shaderProgram.program);

        let projLoc = this.context.getUniformLocation(this._shaderProgram.program, "u_proj");
        let viewLoc = this.context.getUniformLocation(this._shaderProgram.program, "u_view");
        let domainLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.POS_DOMAIN);
        let dimensionLoc = this.context.getUniformLocation(this._shaderProgram.program, ShaderUniforms.DIMENSION);

        this.context.uniformMatrix4fv(projLoc, false, this._camera.projectionMatrix);
        this.context.uniformMatrix4fv(viewLoc, false, this._camera.viewMatrix);
        this.context.uniform4f(domainLoc, textures.getXMin(0), textures.getXMax(0), textures.getYMin(0), textures.getYMax(0));
        this.context.uniform2f(dimensionLoc, textures.width, textures.height);

        if (textures.getTextures(0) != null){

            let id = 0;

            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D_ARRAY, textures.getTextures(0));
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_T0), id++);

            this.context.activeTexture(gl.TEXTURE0 + id);
            this.context.bindTexture(gl.TEXTURE_2D, textures.getMask());
            this.context.uniform1i(this.context.getUniformLocation(this._shaderProgram.program, ShaderElementInputs.TEX_SELECTION), id++);
        }


        this._mesh.draw();
    }

    public currentSelectionChanged(selection: number[]) {
        throw new Error("Method not implemented.");
    }
    public onReset(newValues: TransformableValues) {
        return;
    }
    public onNbElementsChanged(newValues: TransformableValues) {
        return;
    }
    public onNbChannelsChanged(newValues: TransformableValues) {
        return;
    }
    public updateProgamsTransformers(transformers: TransformerBuilder) {
        this._shaderProgram.updateProgramTransformers(transformers.generateTransformersBlock(false, TransformFlag.COLOR));
    }
    public onMouseMoved(deltaX: number, deltaY: number) {
        this._camera.move(-deltaY * this._camera.distance * 0.5, -deltaX * this._camera.distance * 0.5 * this._camera.aspectRatio);
    }
    public onWheelMoved(delta: number) {
        this._camera.moveForward(-delta);
    }
    
    public getViewBoundaries(): [number, number, number, number] {
        let length = ViewerTexture._meshSize / 2;
        return [-length, length, -length, length];
    }
}