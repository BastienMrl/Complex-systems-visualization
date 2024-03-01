import { Viewer } from "../viewer.js";
import { SelectionTool } from "./selectionTool.js";

export class SelectionBrushTool extends SelectionTool{

    private _interactionButton : number;
    
    public constructor(viewer : Viewer, interactionButton : number){
        super(viewer);
        this._interactionButton = interactionButton;
    }

    protected onMouseMove(e: MouseEvent): void {
        
    }
    
    protected onMouseDown(e: MouseEvent): void {
        
    }

    protected onMouseUp(e: MouseEvent): void {
        
    }
}