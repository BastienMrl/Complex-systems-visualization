import { TransformableValues } from "./transformableValues";
import { TransformerBuilder } from "./transformer/transformerBuilder";
import { Viewer } from "./viewer";
import { ViewerManager } from "./viewerManager";

export class ViewerTexture extends Viewer{

    public constructor(canvas : HTMLCanvasElement, context : WebGL2RenderingContext, manager : ViewerManager){
        super(canvas, context, manager);

    }

    public initialization() {
        throw new Error("Method not implemented.");
    }
    public onCanvasResize() {
        throw new Error("Method not implemented.");
    }
    public updateScene(values: TransformableValues) {
        throw new Error("Method not implemented.");
    }
    public clear() {
        throw new Error("Method not implemented.");
    }
    public draw() {
        throw new Error("Method not implemented.");
    }
    public getElementOver(posX: number, posY: number): number {
        throw new Error("Method not implemented.");
    }
    public currentSelectionChanged(selection: number[]) {
        throw new Error("Method not implemented.");
    }
    public onReset(newValues: TransformableValues) {
        throw new Error("Method not implemented.");
    }
    public onNbElementsChanged(newValues: TransformableValues) {
        throw new Error("Method not implemented.");
    }
    public onNbChannelsChanged(newValues: TransformableValues) {
        throw new Error("Method not implemented.");
    }
    public updateProgamsTransformers(transformers: TransformerBuilder) {
        throw new Error("Method not implemented.");
    }
    public onMouseMoved(deltaX: number, deltaY: number) {
        throw new Error("Method not implemented.");
    }
    public onWheelMoved(delta: number) {
        throw new Error("Method not implemented.");
    }
    
}