import { SelectionTool } from "./selectionTool.js";
export class SelectionLassoTool extends SelectionTool {
    _pathIds = [];
    _aabb = [-1, -1, -1, -1];
    _interactionButton;
    constructor(viewer, interactionButton, sizes) {
        super(viewer, sizes);
        this.resetAabb();
        this._interactionButton = interactionButton;
    }
    onMouseMove(e) {
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
    onMouseDown(e) {
        if (e.button != this._interactionButton)
            return;
        this._mouseDown = true;
        this._pathIds = [];
        if (this._currentId != null) {
            this._pathIds.push(this._currentId);
            let coord = this.idToCoords(this._currentId);
            this.extendAabb(coord[0], coord[1]);
        }
    }
    onMouseUp(e) {
        if (e.button != this._interactionButton || !this._mouseDown)
            return;
        this._mouseDown = false;
        this._viewer.sendInteractionRequest(new Float32Array(this._currentMask));
        this.onCurrentSelectionChanged(null);
        this.resetAabb();
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
    setParam(attribute, value) { }
    getAllParam() { return null; }
}
