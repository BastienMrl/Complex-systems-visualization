import { MultipleMeshInstances } from "../mesh.js";
import { StatesTransformer } from "../statesTransformer.js";
import { Viewer } from "../viewer.js";
import { SelectionBoxTool } from "./selectionBox.js";
import { SelectionBrushTool } from "./selectionBrush.js";
import { SelectionLassoTool } from "./selectionLasso.js";
import { SelectionTool } from "./selectionTool.js";

export enum SelectionMode {
    DISABLE = -1,
    BOX = 0,
    BRUSH = 1,
    // Only Convex polygon
    LASSO = 2
}

export class SelectionManager{
    private _mode = SelectionMode.DISABLE;

    private _tools : Array<SelectionTool>;

    constructor(viewer : Viewer){
        this._tools = new Array(3);
        this._tools[SelectionMode.BOX] = new SelectionBoxTool(viewer, 0); 
        this._tools[SelectionMode.BRUSH] = new SelectionBrushTool(viewer, 0);
        this._tools[SelectionMode.LASSO] = new SelectionLassoTool(viewer, 0);


        viewer.canvas.addEventListener('mousemove', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseMoveEvent(e);
        });

        viewer.canvas.addEventListener('mousedown', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseDownEvent(e);
        });

        viewer.canvas.addEventListener('mouseup', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseUpEvent(e);
        });
    }
    
    public switchMode(mode : SelectionMode){
        if (mode == this._mode)
            mode = SelectionMode.DISABLE;
        if (this._mode != SelectionMode.DISABLE){
            this._tools[this._mode].resetTool();
        }
        this._mode = mode;
    }

    public setMeshes(meshes : MultipleMeshInstances){
        this._tools.forEach(e => e.setMeshes(meshes));
    }

    public setTransformer(transformer : StatesTransformer){
        this._tools.forEach(e => e.setTransformer(transformer));
    }
    
}
