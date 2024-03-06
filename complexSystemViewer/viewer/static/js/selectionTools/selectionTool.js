import { Vec3 } from "../ext/glMatrix/vec3.js";
export class SelectionTool {
    _meshes;
    _transformer;
    _viewer;
    mouseX;
    mouseY;
    _currentId;
    _currentMask;
    _mouseDown = false;
    constructor(viewer) {
        this._viewer = viewer;
    }
    // public methods
    setMeshes(meshes) {
        this._meshes = meshes;
        this._currentMask = new Float32Array(this._meshes.nbInstances).fill(0);
    }
    setTransformer(transformer) {
        this._transformer = transformer;
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
        this._viewer.currentSelectionChanged(selection);
        this._currentMask.fill(0.);
        if (selection != null)
            selection.forEach(e => {
                this._currentMask[e] = 1.;
            });
    }
    getMouseOver() {
        let origin = this._viewer.camera.position;
        let x = (2.0 * this.mouseX) / this._viewer.canvas.width - 1.0;
        let y = 1.0 - (2.0 * this.mouseY) / this._viewer.canvas.height;
        let z = 1.0;
        let direction = Vec3.fromValues(x, y, z);
        Vec3.transformMat4(direction, direction, this._viewer.camera.projViewMatrix.invert());
        direction.normalize();
        let normal = Vec3.fromValues(0, 1, 0);
        let pNear = Vec3.fromValues(0, this._meshes.localAabb[3], 0);
        let pFar = Vec3.fromValues(0, this._meshes.localAabb[2], 0);
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
        let offsetX = (this._meshes.nbCol - 1);
        let offsetZ = (this._meshes.nbRow - 1);
        let aabb = this._meshes.localAabb;
        let xMin = (x + aabb[0]) / this._transformer.getPositionFactor(0);
        let xMax = (x + aabb[1]) / this._transformer.getPositionFactor(0);
        let zMin = (z + aabb[4]) / this._transformer.getPositionFactor(2);
        let zMax = (z + aabb[5]) / this._transformer.getPositionFactor(2);
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
        if (z >= this._meshes.nbRow || z < 0 || x >= this._meshes.nbCol || x < 0)
            return null;
        return this.coordToId(z, x);
    }
    coordToId(i, j) {
        return i * this._meshes.nbCol + j;
    }
    idToCoords(id) {
        const nbCol = this._meshes.nbCol;
        const j = id % nbCol;
        const i = (id - j) / nbCol;
        return [i, j];
    }
}
