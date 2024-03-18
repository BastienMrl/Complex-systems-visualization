import { SelectionTool } from "./selectionTool.js";
export class SelectionBoxTool extends SelectionTool {
    _firstId = null;
    _lastId = null;
    _interactionButton;
    constructor(viewer, interactionButton) {
        super(viewer);
        this._interactionButton = interactionButton;
    }
    onMouseMove(e) {
        if (!this._mouseDown)
            return;
        if (this._currentId != null) {
            if (this._firstId == null)
                this._firstId = this._currentId;
            else
                this._lastId = this._currentId;
        }
        if (this._firstId == null) {
            this.onCurrentSelectionChanged(null);
            return;
        }
        let secondId = this._firstId;
        if (this._lastId != null)
            secondId = this._lastId;
        let ids = this.getIdsFromBox(this._firstId, this._lastId != null ? this._lastId : this._firstId);
        this.onCurrentSelectionChanged(ids);
    }
    onMouseUp(e) {
        if (e.button != this._interactionButton || !this._mouseDown)
            return;
        this._mouseDown = false;
        this._viewer.sendInteractionRequest(new Float32Array(this._currentMask));
        this.onCurrentSelectionChanged(null);
    }
    onMouseDown(e) {
        if (e.button != this._interactionButton)
            return;
        this._mouseDown = true;
        this._firstId = this._currentId;
    }
    getIdsFromBox(_firstId, _lastId) {
        let ret = [];
        let firstCoord = this.idToCoords(_firstId);
        let lastCoord = this.idToCoords(_lastId);
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
    setParam(attribute, value) { }
    getAllParam() {
        return [];
    }
}
