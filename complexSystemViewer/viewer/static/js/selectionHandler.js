import * as shaderUtils from "./shaderUtils.js";
// provides access to gl constants
const gl = WebGL2RenderingContext;
export class SelectionHandler {
    //Singleton
    static _instance;
    _context;
    _canvas;
    _frameBuffer;
    _selectionTargetTexture;
    _selectionDepthBuffer;
    _selectionProgram;
    mouseX;
    mouseY;
    selectedId;
    constructor(context) {
        this._context = context;
        this._canvas = this._context.canvas;
        this.selectedId = null;
        this._selectionProgram = new shaderUtils.ProgramWithTransformer(context);
    }
    static getInstance(context) {
        if (!SelectionHandler._instance)
            SelectionHandler._instance = new SelectionHandler(context);
        return SelectionHandler._instance;
    }
    async initialization(srcVs, srcFs) {
        this._canvas.addEventListener('mousemove', (e) => {
            const rect = this._canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });
        await this._selectionProgram.generateProgram(srcVs, srcFs);
        this._frameBuffer = this._context.createFramebuffer();
        this._context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        // this._selectionTargetTexture = this._context.createTexture();
        // this._context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        // this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        // this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        // this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // this._context.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._selectionTargetTexture, 0);
        // this._context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this._canvas.width, this._canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        // this._context.bindTexture(gl.TEXTURE_2D, null);    
        this._selectionTargetTexture = this._context.createRenderbuffer();
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionTargetTexture);
        this._context.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, this._selectionTargetTexture);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, this._canvas.width, this._canvas.height);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, null);
        this._selectionDepthBuffer = this._context.createRenderbuffer();
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this._canvas.width, this._canvas.height);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, null);
        // use this for readPixels and blitFramebuffer
        this._context.readBuffer(gl.COLOR_ATTACHMENT0);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    resizeBuffers() {
        // this._context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        // this._context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this._canvas.width, this._canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this._canvas.width, this._canvas.height);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionTargetTexture);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, this._canvas.width, this._canvas.height);
    }
    updateCurrentSelection(camera, meshes, time) {
        this._context.useProgram(this._selectionProgram.program);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        this._context.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        let projViewLoc = this._context.getUniformLocation(this._selectionProgram.program, "u_proj_view");
        let timeLoc = this._context.getUniformLocation(this._selectionProgram.program, "u_time_translation");
        this._context.uniformMatrix4fv(projViewLoc, false, camera.projViewMatrix);
        this._context.uniform1f(timeLoc, time);
        meshes.drawSelection();
        let data = new Uint8Array(4);
        this._context.readPixels(this.mouseX, this._canvas.height - this.mouseY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, data);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, null);
        let id = data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24);
        if (id != this.selectedId) {
            this.selectedId = id > 0 ? id : null;
        }
        console.log("id = ", id);
    }
    updateProgamTransformers(transformers) {
        this._selectionProgram.updateProgramTransformers(transformers.generateTranslationTransformersBlock());
    }
}
