import { Viewer } from "./viewer.js";
export class ViewerTexture extends Viewer {
    constructor(canvas, context, manager) {
        super(canvas, context, manager);
    }
    initialization() {
        // throw new Error("Method not implemented.");
        return;
    }
    onCanvasResize() {
        throw new Error("Method not implemented.");
    }
    updateScene(values) {
        throw new Error("Method not implemented.");
    }
    clear() {
        throw new Error("Method not implemented.");
    }
    draw() {
        throw new Error("Method not implemented.");
    }
    getElementOver(posX, posY) {
        throw new Error("Method not implemented.");
    }
    currentSelectionChanged(selection) {
        throw new Error("Method not implemented.");
    }
    onReset(newValues) {
        throw new Error("Method not implemented.");
    }
    onNbElementsChanged(newValues) {
        throw new Error("Method not implemented.");
    }
    onNbChannelsChanged(newValues) {
        throw new Error("Method not implemented.");
    }
    updateProgamsTransformers(transformers) {
        throw new Error("Method not implemented.");
    }
    onMouseMoved(deltaX, deltaY) {
        throw new Error("Method not implemented.");
    }
    onWheelMoved(delta) {
        throw new Error("Method not implemented.");
    }
}
