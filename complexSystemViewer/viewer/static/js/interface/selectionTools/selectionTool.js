export class SelectionTool {
    _viewer;
    mouseX;
    mouseY;
    _currentId;
    _currentMask;
    _mouseDown = false;
    constructor(viewer) {
        this._viewer = viewer;
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
            this._viewer.currentSelectionChanged(selection);
            if (selection != null)
                selection.forEach(e => {
                    this._currentMask[e] = 1.;
                });
        }
        else if (selection instanceof Map) {
            this._viewer.currentSelectionChanged(Array.from(selection.keys()));
            selection.forEach((value, key) => {
                this._currentMask[key] = value;
            });
        }
        else {
            this._viewer.currentSelectionChanged(null);
        }
    }
    getMouseOver() {
        return this._viewer.getElementOver(this.mouseX, this.mouseY);
    }
    coordToId(i, j) {
        return 0;
        // return i * this._meshes.nbCol + j;
    }
    idToCoords(id) {
        return [0, 0];
        // const nbCol = this._meshes.nbCol;
        // const j = id % nbCol;
        // const i = (id - j) / nbCol
        // return [i, j] 
    }
}
