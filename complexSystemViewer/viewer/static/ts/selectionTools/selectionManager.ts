import { MultipleMeshInstances } from "../mesh.js";
import { Stats } from "../stats.js";
import { TransformerBuilder } from "../transformerBuilder.js";
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
    
    private _stats : Stats;

    constructor(viewer : Viewer, stats : Stats){
        this._tools = new Array(3);
        this._tools[SelectionMode.BOX] = new SelectionBoxTool(viewer, 0); 
        this._tools[SelectionMode.BRUSH] = new SelectionBrushTool(viewer, 0);
        this._tools[SelectionMode.LASSO] = new SelectionLassoTool(viewer, 0);

        this._stats = stats;


        viewer.canvas.addEventListener('mousemove', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseMoveEvent(e);
        });

        viewer.canvas.addEventListener('mousedown', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE)
                this._tools[this._mode].receiveMouseDownEvent(e);
        });

        viewer.canvas.addEventListener('mouseup', (e : MouseEvent) => {
            if (this._mode != SelectionMode.DISABLE){
                this._stats.startPickingTimer();
                this._tools[this._mode].receiveMouseUpEvent(e);
                this._stats.stopPickingTimer();
            }
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

    public setTransformer(transformer : TransformerBuilder){
        this._tools.forEach(e => e.setTransformer(transformer));
    }

    public setSelectionParameter(attribute:string, value:number){
        this._tools[this._mode].setParam(attribute,value);
    }

    public getSelectionParameter():string{
        return this._tools[this._mode].getAllParam();
    }
    
}
