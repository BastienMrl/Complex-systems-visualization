export class Stats{
    private static _nbDigits = 1;

    private _fpsEl : HTMLElement;
    private _fpsAccumulator : number = 0;

    private _renderingEl : HTMLElement;
    private _renderingTimer : number;
    private _nbRendering : number = 0;
    private _renderingAccumulator : number = 0;
    private _currentRenderingDelay : number = 0;


    private _updateEl : HTMLElement;
    private _updateTimer : number;
    private _nbUpdates : number = 0;
    private _updateAccumulator : number = 0;
    private _currentUpdateDelay : number = 0;

    private _pickingEl : HTMLElement;
    private _pickingTimer : number;
    private _nbPicking : number = 0;
    private _pickingAccumulator : number = 0;
    private _currentPickingDelay : number = 0;

    private _totalEl : HTMLElement;

    private _nbIteration : number = 10;

    public constructor(fpsEl : HTMLElement, updateEl : HTMLElement, renderingEl : HTMLElement, pickingEl : HTMLElement, totalEl : HTMLElement){
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
        this._pickingEl = pickingEl;
        this._totalEl = totalEl;
    }
    
    private displayFPS(fps : number){
        this._fpsEl.innerHTML = "FPS : " + fps.toFixed(0);
        const total =  this._currentRenderingDelay + this._currentPickingDelay + this._currentUpdateDelay;
        this._totalEl.innerHTML = "Total : " + total.toFixed(Stats._nbDigits) + " ms";
    }

    private displayRendering(delay : number){
        this._renderingEl.innerHTML = "Rendering : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentRenderingDelay = delay;
    }

    private displayUpdate(delay : number){
        this._updateEl.innerHTML = "Update : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentUpdateDelay = delay;
    }

    private displayPicking(delay : number){
        this._pickingEl.innerHTML = "Picking : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentPickingDelay = delay;
    }

    public startRenderingTimer(delta : number){
        this._renderingTimer = performance.now();
        this._fpsAccumulator += delta;
        this._nbRendering += 1;
        if (this._nbRendering == this._nbIteration){
            const delay : number = this._fpsAccumulator / this._nbIteration;
            this._fpsAccumulator = 0;
            this.displayFPS(Math.round(1. / delay));
        }
    }
    
    public stopRenderingTimer(){
        const delta : number = performance.now() - this._renderingTimer
        this._renderingAccumulator += delta
        if (this._nbRendering == this._nbIteration){
            const delay : number = this._renderingAccumulator / this._nbIteration;
            this._nbRendering = 0;
            this._renderingAccumulator = 0;
            this.displayRendering(delay);
        }
    }

    public startUpdateTimer(){
        this._updateTimer = performance.now();
    }

    public stopUpdateTimer(){
        const delta : number = performance.now() - this._updateTimer
        this._updateAccumulator += delta;
        this._nbUpdates += 1;
        if (this._nbUpdates == this._nbIteration){
            const delay : number = this._updateAccumulator / this._nbIteration;
            this._nbUpdates = 0;
            this._updateAccumulator = 0;
            this.displayUpdate(delay);
        }
    }

    public startPickingTimer(){
        this._pickingTimer = performance.now();
    }

    public stopPickingTimer(){
        const delta : number = performance.now() - this._pickingTimer
        this._pickingAccumulator += delta;
        this._nbPicking += 1;
        if (this._nbPicking == this._nbIteration){
            const delay : number = this._pickingAccumulator / this._nbIteration;
            this._nbPicking = 0;
            this._pickingAccumulator = 0;
            this.displayPicking(delay);
        }
    }
}