import { SelectionTool } from "./selectionTool.js";
export class SelectionBrushTool extends SelectionTool {
    _interactionButton;
    constructor(viewer, interactionButton) {
        super(viewer);
        this._interactionButton = interactionButton;
    }
    onMouseMove(e) {
    }
    onMouseDown(e) {
    }
    onMouseUp(e) {
    }
}
