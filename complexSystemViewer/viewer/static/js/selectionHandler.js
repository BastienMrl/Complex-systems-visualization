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
        this._selectionProgram = await shaderUtils.initShaders(this._context, srcVs, srcFs);
        this._frameBuffer = this._context.createFramebuffer();
        this._context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        this._selectionTargetTexture = this._context.createTexture();
        this._context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        this._context.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this._context.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._selectionTargetTexture, 0);
        this._context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this._canvas.width, this._canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        this._context.bindTexture(gl.TEXTURE_2D, null);
        this._selectionDepthBuffer = this._context.createRenderbuffer();
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this._canvas.width, this._canvas.height);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, null);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    resizeBuffers() {
        this._context.bindTexture(gl.TEXTURE_2D, this._selectionTargetTexture);
        this._context.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this._canvas.width, this._canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        this._context.bindRenderbuffer(gl.RENDERBUFFER, this._selectionDepthBuffer);
        this._context.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this._canvas.width, this._canvas.height);
    }
    updateCurrentSelection(camera, meshes, time) {
        this._context.useProgram(this._selectionProgram);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, this._frameBuffer);
        this._context.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        let projViewLoc = this._context.getUniformLocation(this._selectionProgram, "u_proj_view");
        let timeLoc = this._context.getUniformLocation(this._selectionProgram, "u_time");
        this._context.uniformMatrix4fv(projViewLoc, false, camera.projViewMatrix);
        this._context.uniform1f(timeLoc, time);
        meshes.drawSelection();
        let data = new Uint8Array(4);
        this._context.readPixels(this.mouseX, this._canvas.height - this.mouseY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, data);
        this._context.bindFramebuffer(gl.FRAMEBUFFER, null);
        let id = data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24) - 1;
        if (id != this.selectedId) {
            this.selectedId = id >= 0 ? id : null;
        }
    }
    hasCurrentSelection() {
        return this.selectedId != null;
    }
}