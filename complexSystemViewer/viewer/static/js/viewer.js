export class Viewer {
    context;
    canvas;
    _manager;
    _camera;
    _selectionManager;
    _animationTimer;
    _transmissionWorker;
    _isDrawable;
    constructor(canvas, context, manager) {
        this.context = context;
        this.canvas = canvas;
        this._manager = manager;
        this._isDrawable = false;
    }
    get selectionManager() {
        return this._selectionManager;
    }
    get transmissionWorker() {
        return this._transmissionWorker;
    }
    get isDrawable() {
        return this._isDrawable;
    }
    set isDrawable(value) {
        this._isDrawable = value;
    }
    get camera() {
        return this._camera;
    }
}
