import { Vec3 } from "./ext/glMatrix/vec3.js";
export var PickingMode;
(function (PickingMode) {
    PickingMode[PickingMode["DISABLE"] = 0] = "DISABLE";
    PickingMode[PickingMode["POINT"] = 1] = "POINT";
    PickingMode[PickingMode["BOX"] = 2] = "BOX";
    // Only Convex polygon
    PickingMode[PickingMode["LASSO"] = 3] = "LASSO";
})(PickingMode || (PickingMode = {}));
export class PickingTool {
    _meshes;
    _transformer;
    _viewer;
    _mode;
    mouseX;
    mouseY;
    _currentId;
    _mouseDown = false;
    _pathIds = [];
    _aabb = [-1, -1, -1, -1];
    _currentMask;
    constructor(viewer) {
        this._viewer = viewer;
        this._viewer.canvas.addEventListener('mousemove', (e) => {
            this.onMouseMove(e);
        });
        this.resetAabb();
        this._viewer.canvas.addEventListener('mousedown', (e) => {
            switch (this._mode) {
                case PickingMode.BOX:
                    this.onMouseClickBox(e);
                    break;
                case PickingMode.LASSO:
                    this.onMouseClickLasso(e);
                    break;
            }
        });
        this._viewer.canvas.addEventListener('mouseup', (e) => {
            switch (this._mode) {
                case PickingMode.POINT:
                    this.onMouseReleasePoint(e);
                    break;
                case PickingMode.BOX:
                    this.onMouseReleaseBox(e);
                    break;
                case PickingMode.LASSO:
                    this.onMouseReleaseLasso(e);
                    break;
            }
            this._viewer.sendInteractionRequest(new Float32Array(this._currentMask));
            this.onCurrentSelectionChanged(null);
        });
        window.addEventListener('keydown', (e) => {
            if (e.key == "Enter")
                this._viewer.sendInteractionRequest(new Float32Array(this._currentMask));
        });
        this._mode = PickingMode.DISABLE;
    }
    setMeshes(meshes) {
        this._meshes = meshes;
        this._currentMask = new Float32Array(this._meshes.nbInstances).fill(0);
    }
    setTransformer(transformer) {
        this._transformer = transformer;
    }
    switMode(mode) {
        if (mode == this._mode)
            mode = PickingMode.DISABLE;
        this._mode = mode;
        this._mouseDown = false;
        this._viewer.currentSelectionChanged(null);
    }
    onMouseMove(e) {
        if (this._mode == PickingMode.DISABLE)
            return;
        const rect = this._viewer.canvas.getBoundingClientRect();
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
        let t = performance.now();
        this._currentId = this.getMouseOver();
        t = performance.now() - t;
        // console.log("time = ", t, "ms");
        switch (this._mode) {
            case PickingMode.BOX:
                {
                    if (!this._mouseDown)
                        break;
                    if (this._currentId != null) {
                        if (this._pathIds[0] == undefined || this._pathIds[0] == null)
                            this._pathIds[0] = this._currentId;
                        else
                            this._pathIds[1] = this._currentId;
                    }
                    if (this._pathIds[0] == undefined || this._pathIds[0] == null) {
                        this.onCurrentSelectionChanged(null);
                        return;
                    }
                    let secondId = this._pathIds[0];
                    if (this._pathIds[1] != undefined && this._pathIds[1] != null)
                        secondId = this._pathIds[1];
                    let ids = this.getIdsFromBox(this._pathIds[0], secondId);
                    this.onCurrentSelectionChanged(ids);
                }
                break;
            case PickingMode.LASSO:
                {
                    if (!this._mouseDown)
                        return;
                    if (this._currentId != null) {
                        let coord = this.idToCoords(this._currentId);
                        if (this.extendAabb(coord[0], coord[1]))
                            this.addLineToPath(this._currentId);
                    }
                    let ret = this.getIdsInsideLasso();
                    if (ret.length == 0)
                        ret = null;
                    this.onCurrentSelectionChanged(ret);
                }
                break;
        }
    }
    onMouseClickBox(e) {
        if (e.button != 0)
            return;
        this._mouseDown = true;
        this._pathIds = [this._currentId];
    }
    onMouseClickLasso(e) {
        if (e.button != 0)
            return;
        this._mouseDown = true;
        this._pathIds = [];
        if (this._currentId != null) {
            this._pathIds.push(this._currentId);
            let coord = this.idToCoords(this._currentId);
            this.extendAabb(coord[0], coord[1]);
        }
    }
    onMouseReleasePoint(e) {
        if (e.button != 0)
            return;
        this.onCurrentSelectionChanged(this._currentId == null ? null : [this._currentId]);
    }
    onMouseReleaseBox(e) {
        if (e.button != 0 || !this._mouseDown)
            return;
        this._mouseDown = false;
    }
    onMouseReleaseLasso(e) {
        if (e.button != 0 || !this._mouseDown)
            return;
        this._mouseDown = false;
        this.resetAabb();
    }
    onCurrentSelectionChanged(selection) {
        this._viewer.currentSelectionChanged(selection);
        this._currentMask.fill(0.);
        if (selection != null)
            selection.forEach(e => {
                this._currentMask[e] = 1.;
            });
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
    resetAabb() {
        this._aabb[0] = Number.MAX_SAFE_INTEGER;
        this._aabb[1] = Number.MIN_SAFE_INTEGER;
        this._aabb[2] = Number.MAX_SAFE_INTEGER;
        this._aabb[3] = Number.MIN_SAFE_INTEGER;
    }
    extendAabb(i, j) {
        let ret = false;
        if (i < this._aabb[0]) {
            this._aabb[0] = i;
            ret = true;
        }
        if (i > this._aabb[1]) {
            this._aabb[1] = i;
            ret = true;
        }
        if (j < this._aabb[2]) {
            this._aabb[2] = j;
            ret = true;
        }
        if (j > this._aabb[3]) {
            this._aabb[3] = j;
            ret = true;
        }
        return ret;
    }
    getIdsInsideLasso() {
        let oldPath = this._pathIds.slice();
        this.addLineToPath(this._pathIds[0]);
        this._pathIds.pop();
        let ret = [];
        for (let i = this._aabb[0] + 1; i < this._aabb[1]; i++) {
            for (let j = this._aabb[2] + 1; j < this._aabb[3]; j++) {
                let left = 0;
                let right = 0;
                this._pathIds.forEach((id) => {
                    const coord = this.idToCoords(id);
                    const iTest = coord[0];
                    const jTest = coord[1];
                    if (i == iTest) {
                        if (j <= jTest)
                            right++;
                        if (j >= jTest)
                            left++;
                    }
                });
                if (left != 0 && right != 0)
                    ret.push(this.coordToId(i, j));
            }
        }
        this._pathIds = oldPath;
        return ret.concat(this._pathIds);
    }
    // Bresenham's line algorithm from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    addLineToPath(targetId) {
        if (targetId == null || !Number.isInteger(targetId))
            return;
        if (this._pathIds.length == 0) {
            this._pathIds.push(targetId);
            return;
        }
        const targetCoord = this.idToCoords(targetId);
        const fromCoord = this.idToCoords(this._pathIds.pop());
        let i0 = fromCoord[0];
        const i1 = targetCoord[0];
        let j0 = fromCoord[1];
        const j1 = targetCoord[1];
        const di = Math.abs(i1 - i0);
        const si = i0 < i1 ? 1 : -1;
        const dj = -Math.abs(j1 - j0);
        const sj = j0 < j1 ? 1 : -1;
        let error = di + dj;
        while (true) {
            this._pathIds.push(this.coordToId(i0, j0));
            if (i0 == i1 && j0 == j1)
                break;
            const e2 = 2 * error;
            if (e2 > dj) {
                error += dj;
                i0 += si;
            }
            if (e2 < di) {
                error += di;
                j0 += sj;
            }
        }
    }
    getIdsFromBox(firstId, lastId) {
        let ret = [];
        let firstCoord = this.idToCoords(firstId);
        let lastCoord = this.idToCoords(lastId);
        let firstIsImin = firstCoord[0] < lastCoord[0];
        let firstIsJmin = firstCoord[1] < lastCoord[1];
        let iMin = firstIsImin ? firstCoord[0] : lastCoord[0];
        let iMax = firstIsImin ? lastCoord[0] : firstCoord[0];
        let jMin = firstIsJmin ? firstCoord[1] : lastCoord[1];
        let jMax = firstIsJmin ? lastCoord[1] : firstCoord[1];
        for (let i = iMin; i < iMax + 1; i++)
            for (let j = jMin; j < jMax + 1; j++)
                ret.push(this.coordToId(i, j));
        return ret;
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
}
