import { SelectionBoxTool } from "./selectionBox.js";
import { SelectionBrushTool } from "./selectionBrush.js";
import { SelectionLassoTool } from "./selectionLasso.js";
export var SelectionMode;
(function (SelectionMode) {
    SelectionMode[SelectionMode["DISABLE"] = -1] = "DISABLE";
    SelectionMode[SelectionMode["BOX"] = 0] = "BOX";
    SelectionMode[SelectionMode["BRUSH"] = 1] = "BRUSH";
    // Only Convex polygon
    SelectionMode[SelectionMode["LASSO"] = 2] = "LASSO";
})(SelectionMode || (SelectionMode = {}));
export class SelectionManager {
    _mode = SelectionMode.DISABLE;
    _tools;
    _stats;
    constructor(viewer, stats) {
        this._tools = new Array(3);
        this._tools[SelectionMode.BOX] = new SelectionBoxTool(viewer, 0);
        this._tools[SelectionMode.BRUSH] = new SelectionBrushTool(viewer, 0);
        this._tools[SelectionMode.LASSO] = new SelectionLassoTool(viewer, 0);
        this._stats = stats;
        viewer.canvas.addEventListener('mousemove', (e) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseMoveEvent(e);
        });
        viewer.canvas.addEventListener('mousedown', (e) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseDownEvent(e);
        });
        viewer.canvas.addEventListener('mouseup', (e) => {
            if (this._mode != SelectionMode.DISABLE) {
                this._stats.startPickingTimer();
                this._tools[this._mode].receiveMouseUpEvent(e);
                this._stats.stopPickingTimer();
            }
        });
    }
    switchMode(mode) {
        if (mode == this._mode)
            mode = SelectionMode.DISABLE;
        if (this._mode != SelectionMode.DISABLE) {
            this._tools[this._mode].resetTool();
        }
        this._mode = mode;
    }
    setMeshes(meshes) {
        this._tools.forEach(e => e.setMeshes(meshes));
    }
    setTransformer(transformer) {
        this._tools.forEach(e => e.setTransformer(transformer));
    }
}
