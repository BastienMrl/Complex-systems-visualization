import { AnimationTimer } from "./animationTimer";
import { SelectionManager } from "./interface/selectionTools/selectionManager";
import { TransformableValues } from "./transformableValues";
import { TransformerBuilder } from "./transformer/transformerBuilder";
import { TexturesContainer, ViewerManager } from "./viewerManager";

export abstract class Viewer{
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    protected _manager : ViewerManager;

    

    protected _selectionManager : SelectionManager

    protected _animationTimer : AnimationTimer;
    protected _transmissionWorker : Worker;

    private _isDrawable : boolean;

    public constructor(canvas : HTMLCanvasElement, context : WebGL2RenderingContext, manager : ViewerManager){
        this.context = context;
        this.canvas = canvas;
        this._manager = manager;
        this._isDrawable = false;
    }


    public get selectionManager() : SelectionManager {
        return this._selectionManager;
    }

    public get transmissionWorker() : Worker {
        return this._transmissionWorker;
    }

    public get isDrawable() : boolean {
        return this._isDrawable;
    }

    public set isDrawable(value : boolean) {
        this._isDrawable = value;
    }


    // initialization methods
    public abstract initialization();

    public abstract onCanvasResize();

    public abstract updateScene(values : TransformableValues);

    public abstract clear();

    public abstract draw(textures : TexturesContainer);

    public abstract getElementOver(posX : number, posY : number) : number | null;
    
    public abstract currentSelectionChanged(selection : Array<number> | null);

    public abstract onReset(newValues : TransformableValues);

    public abstract onNbElementsChanged(newValues : TransformableValues);

    public abstract onNbChannelsChanged(newValues : TransformableValues);

    public abstract updateProgamsTransformers(transformers : TransformerBuilder)

    public abstract onMouseMoved(deltaX : number, deltaY : number);

    public abstract onWheelMoved(delta : number);

}
