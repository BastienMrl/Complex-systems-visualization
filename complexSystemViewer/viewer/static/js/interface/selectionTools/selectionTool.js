import { Vec3 } from "../../ext/glMatrix/vec3.js";
export class SelectionTool {
    _viewer;
    mouseX;
    mouseY;
    _maskSize;
    _currentId;
    _currentMask;
    _mouseDown = false;
    _manager;
    constructor(viewer, manager) {
        this._viewer = viewer;
        this._manager = manager;
        this._maskSize = manager.maskSize;
        this._currentMask = new Float32Array(this._maskSize[0] * this._maskSize[1]);
    }
    resetTool() {
        this._mouseDown = false;
        this.onCurrentSelectionChanged(null);
    }
    receiveMouseMoveEvent(e) {
        const rect = this._viewer.canvas.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
        this._currentId = this.getMouseOver();
        this.onMouseMove(e);
    }
    receiveMouseDownEvent(e) {
        this.onMouseDown(e);
    }
    receiveMouseUpEvent(e) {
        this.onMouseUp(e);
    }
    // protected methods
    onCurrentSelectionChanged(selection) {
        this._currentMask.fill(-1);
        if (selection instanceof Array) {
            if (selection != null)
                selection.forEach(e => {
                    this._currentMask[e] = 1.;
                });
        }
        else if (selection instanceof Map) {
            selection.forEach((value, key) => {
                this._currentMask[key] = value;
            });
        }
        this._viewer.updateMaskTexture(this._currentMask);
    }
    getMouseOver() {
        let boundaries = this._viewer.getViewBoundaries();
        let origin = Vec3.create();
        let x = null;
        let y = null;
        let z = null;
        let direction = Vec3.create();
        if (this._viewer.camera.isOrthographic) {
            let boundaries = this._viewer.camera.getOrthographicBoundaries();
            x = (boundaries[1] - boundaries[0]) * 0.5 * ((2.0 * this.mouseX) / this._viewer.canvas.width - 1.0);
            y = this._viewer.camera.position.y;
            z = (boundaries[2] - boundaries[3]) * 0.5 * (1.0 - (2.0 * this.mouseY) / this._viewer.canvas.height);
            origin = Vec3.fromValues(x, y, z).add(this._viewer.camera.position);
            Vec3.sub(direction, this._viewer.camera.target, this._viewer.camera.position);
            direction.normalize();
        }
        else {
            // ray initialization
            origin = this._viewer.camera.position;
            x = (2.0 * this.mouseX) / this._viewer.canvas.width - 1.0;
            y = 1.0 - (2.0 * this.mouseY) / this._viewer.canvas.height;
            z = 1.0;
            direction = Vec3.fromValues(x, y, z);
            Vec3.transformMat4(direction, direction, this._viewer.camera.projViewMatrix.invert());
            direction.normalize();
        }
        let normal = Vec3.fromValues(0, 1, 0);
        let denominator = Vec3.dot(normal, direction);
        let t = null;
        if (denominator != 0) {
            let p = Vec3.create();
            Vec3.sub(p, Vec3.fromValues(0, 0, 0), origin);
            t = Vec3.dot(p, normal) / denominator;
        }
        if (t == null || t < 0) {
            return null;
        }
        let position = Vec3.create();
        let dir = Vec3.create();
        Vec3.copy(dir, direction);
        dir.scale(t);
        Vec3.add(position, origin, dir);
        if (position.x < boundaries[0] || position.x > boundaries[1] || position.z < boundaries[2] || position.z > boundaries[3]) {
            return null;
        }
        let map = function (value, fromMin, fromMax, toMin, toMax) {
            return toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
        };
        let j = Math.round(map(position.x, boundaries[0], boundaries[1], 0, 1) * this._maskSize[0] - 1);
        let i = Math.round(map(position.z, boundaries[2], boundaries[3], 0, 1) * this._maskSize[1] - 1);
        return this.coordToId(i, j);
    }
    coordToId(i, j) {
        return i * this._maskSize[0] + j;
    }
    idToCoords(id) {
        const nbCol = this._maskSize[0];
        const j = id % nbCol;
        const i = (id - j) / nbCol;
        return [i, j];
    }
}
