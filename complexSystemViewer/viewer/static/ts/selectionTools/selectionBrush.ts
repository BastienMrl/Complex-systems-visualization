import { Viewer } from "../viewer.js";
import { SelectionTool } from "./selectionTool.js";

export enum BrushShape{
    SQUARE,
    CIRCLE
}

export class SelectionBrushTool extends SelectionTool{
    private static readonly _radiusMin = 0;
    private static readonly _radiusMax = 30;

    private _interactionButton : number;
    private _shape : BrushShape;
    private _radius : number;
    private _attenuationFunction : (distance : number) => number

    private _prevId : number | null;
    private _idValues : Map<number, number>;

    
    public constructor(viewer : Viewer, interactionButton : number){
        super(viewer);
        this._interactionButton = interactionButton;
        this._shape = BrushShape.CIRCLE;
        this._radius = 3;
        this._attenuationFunction = function (distance : number){
            if (distance > this._radius)
                return 0;
            let normalized = distance / (this._radius + 1);
            return 1 - normalized;
        }
        this._prevId = null;
        this._idValues = new Map<number, number>();

        // Temporary
        window.addEventListener('keydown', e => {
            if (e.code == 'ArrowUp')
                this.setBrushRadius(this._radius + 1);
            else if (e.code == 'ArrowDown') 
                this.setBrushRadius(this._radius - 1);
            else if (e.code == 'ArrowLeft' || e.code == 'ArrowRight'){
                let shape = this._shape == BrushShape.CIRCLE ? BrushShape.SQUARE : BrushShape.CIRCLE;
                this.setBrushShape(shape);
            }

        });
    }

    public setBrushRadius(radius : number){
        if (radius < SelectionBrushTool._radiusMin)
            this._radius = SelectionBrushTool._radiusMin;
        else if (radius > SelectionBrushTool._radiusMax)
            this._radius = SelectionBrushTool._radiusMax;
        else
            this._radius = radius;
        this.updateCurrentMouseOver();
    }

    public setBrushShape(shape : BrushShape){
        this._shape = shape;
        this.updateCurrentMouseOver();
    }

    protected onMouseMove(e: MouseEvent): void {
        if (!this._mouseDown){
            this.updateCurrentMouseOver();
            return;
        }
        
        else if (this._currentId != null){
            let path = [this._currentId];
            if (this._prevId != null)
                path = this.getPath(this._prevId, this._currentId);
            this._prevId = this._currentId;
            this.fillCurrentValues(path);
        }
        else{
            this._prevId = null;
        }
        this.onCurrentSelectionChanged(Array.from(this._idValues.keys()));
    }

    protected updateCurrentMouseOver(){
        if (this._mouseDown)
            return;
        this._idValues.clear();

        if (this._currentId == null)
            this.onCurrentSelectionChanged(null);
        else{
            this.fillCurrentValues([this._currentId]);
            this.onCurrentSelectionChanged(Array.from(this._idValues.keys()));
        }
    }
    
    protected onMouseDown(e: MouseEvent): void {
        if (e.button != this._interactionButton)
            return;

        this._mouseDown = true;
        this._prevId = this._currentId;
        this._idValues.clear();
    }

    protected onMouseUp(e: MouseEvent): void {
        if (e.button != this._interactionButton || !this._mouseDown)
            return;
        this._mouseDown = false;
        if (this._currentMask.length != 0)
            this._viewer.sendInteractionRequest(new Float32Array(this._currentMask));
        this.onCurrentSelectionChanged(null);
    }

    private getPath(fromId : number, targetId : number) : Array<number>{
        if (fromId == null || !Number.isInteger(fromId))
            return [];
        if (targetId == null || !Number.isInteger(targetId))
            return [];

        const targetCoord = this.idToCoords(targetId);
        const fromCoord = this.idToCoords(fromId)

        let ret : number[] = [];

        let i0 = fromCoord[0];
        const i1 = targetCoord[0];
        let j0 = fromCoord[1];
        const j1 = targetCoord[1];

        const di = Math.abs(i1 - i0);
        const si = i0 < i1 ? 1 : -1;
        const dj = - Math.abs(j1 - j0);
        const sj = j0 < j1 ? 1 : -1;
        let error = di + dj;

        while (true){
            ret.push(this.coordToId(i0, j0));
            if (i0 == i1 && j0 == j1)
                break;

            const e2 = 2 * error;
            if (e2 > dj) { error += dj; i0 += si; }
            if (e2 <  di) { error += di; j0 += sj; }
        }

        return ret;
    }

    private fillCurrentValues(path : Array<number>){
        path.forEach((e) => {
            let newMap = this.getIdValuesInsideBrush(e);
            newMap.forEach((value, key, map) => {
                let currentValue = this._idValues.get(key);
                if (currentValue == undefined || currentValue < value)
                    this._idValues.set(key, value);
            });
        });
    }

    private getIdValuesInsideBrush(from : number) : Map<number, number>{

        let ret = new Map<number, number>();
        
        let centerCoord = this.idToCoords(from);
        
        let iMin = centerCoord[0] - this._radius;
        let iMax = centerCoord[0] + this._radius;
        let jMin = centerCoord[1] - this._radius; 
        let jMax = centerCoord[1] + this._radius;

        if (iMin < 0)
            iMin = 0;
        if (iMax >= this._meshes.nbRow)
            iMax = this._meshes.nbRow - 1;
        if (jMin < 0)
            jMin = 0;
        if (jMax >= this._meshes.nbCol)
            jMax = this._meshes.nbCol - 1;

        
        for(let i = iMin; i < iMax + 1; i++){
            for (let j = jMin; j < jMax + 1; j++){
                let absI = Math.abs(centerCoord[0] - i);
                let absJ = Math.abs(centerCoord[1] - j);
                // attenuationFunction exclude this case
                let distance = this._radius + 1;
                switch (this._shape){
                    case BrushShape.SQUARE:
                        distance = Math.min(absI, absJ);
                        break;
                    case BrushShape.CIRCLE:
                        distance = Math.sqrt(absI * absI + absJ * absJ);
                        break;
                }
                if (distance <= this._radius)
                    ret.set(this.coordToId(i, j), this._attenuationFunction(distance));                
            }
        }

        return ret;
    }
}