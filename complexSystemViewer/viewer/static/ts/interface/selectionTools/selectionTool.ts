import { ViewerManager } from "../../viewerManager.js";

export abstract class SelectionTool{
    protected _viewer : ViewerManager;

    public mouseX : number;
    public mouseY : number;

    protected _currentId : number | null;

    protected _currentMask : Float32Array;

    protected _mouseDown : boolean = false;



    public constructor(viewer : ViewerManager){
        this._viewer = viewer;
    }

    protected abstract onMouseMove(e : MouseEvent) : void;
    protected abstract onMouseDown(e : MouseEvent) : void;
    protected abstract onMouseUp(e : MouseEvent) : void;


    // public methods
    public abstract setParam(attribute:string, value:number) : void;
    public abstract getAllParam() : string;

    public resetTool(){
        this._mouseDown = false;
        this.onCurrentSelectionChanged(null)
    }

    public receiveMouseMoveEvent(e : MouseEvent) : void {
        const rect = this._viewer.canvas.getBoundingClientRect();   
        this.mouseX = e.clientX - rect.left;
        this.mouseY = e.clientY - rect.top;
        this._currentId = this.getMouseOver();
        this.onMouseMove(e);
    }

    public receiveMouseDownEvent(e : MouseEvent) : void {
        this.onMouseDown(e);
    }

    public receiveMouseUpEvent(e : MouseEvent) : void {
        this.onMouseUp(e);
    }


    // protected methods
    protected onCurrentSelectionChanged(selection : Array<number> | Map<number, number> | null){
        this._currentMask.fill(-1);
        if (selection instanceof Array){

            this._viewer.currentSelectionChanged(selection);
            if (selection != null)
                selection.forEach(e => {
                    this._currentMask[e] = 1.;
            });
        }
        else if (selection instanceof Map){
            this._viewer.currentSelectionChanged(Array.from(selection.keys()));
            selection.forEach((value, key) =>{
                this._currentMask[key] = value;
            });
        }
        else {
            this._viewer.currentSelectionChanged(null);
        }
    }



    protected getMouseOver(){
        return this._viewer.getElementOver(this.mouseX, this.mouseY);
    }

    protected coordToId(i : number, j : number) : number{
        return 0;
        // return i * this._meshes.nbCol + j;
    }

    protected idToCoords(id : number) : [number, number]{
        return [0, 0]
        // const nbCol = this._meshes.nbCol;
        // const j = id % nbCol;
        // const i = (id - j) / nbCol
        // return [i, j] 
    }
}