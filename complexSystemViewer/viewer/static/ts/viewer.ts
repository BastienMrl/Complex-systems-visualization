import { AnimationTimer } from "./animationTimer";
import { Camera } from "./camera.js";
import { SelectionManager } from "./interface/selectionTools/selectionManager";
import { TransformableValues } from "./transformableValues";
import { TransformerBuilder } from "./transformer/transformerBuilder";
import { TexturesContainer, ViewerManager } from "./viewerManager";

export abstract class Viewer{
    public context : WebGL2RenderingContext;
    public canvas : HTMLCanvasElement;
    protected _manager : ViewerManager;

    protected _camera : Camera;

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

    public get camera() : Camera{
        return this._camera
    }

    // initialization methods
    public abstract initialization();

    public abstract onCanvasResize();

    public abstract updateScene(values : TransformableValues);

    public abstract clear();

    public abstract draw(textures : TexturesContainer);
    
    public abstract currentSelectionChanged(selection : Array<number> | null);

    public abstract onReset(newValues : TransformableValues);

    public abstract onNbElementsChanged(newValues : TransformableValues);

    public abstract onNbChannelsChanged(newValues : TransformableValues);

    public abstract updateProgamsTransformers(transformers : TransformerBuilder)

    public abstract onMouseMoved(deltaX : number, deltaY : number);

    public abstract onWheelMoved(delta : number);

    public abstract getViewBoundaries() : [number, number, number, number];

    protected abstract initCamera();
}
