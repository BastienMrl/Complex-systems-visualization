export class Stats{

    private _fpsEl : HTMLElement;
    private _updateEl : HTMLElement;
    private _renderingEl : HTMLElement;

    private _renderingTimer : number;
    private _updateTimer : number;

    private _nbUpdates : number = 0
    private _nbRendering : number = 0

    private _fpsAccumulator : number = 0;
    private _renderingAccumulator : number = 0;
    private _updateAccumulator : number = 0;

    private _nbIteration : number = 10;

    public constructor(fpsEl : HTMLElement, updateEl : HTMLElement, renderingEl : HTMLElement){
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
    }
    
    private displayFPS(fps : number){
        this._fpsEl.innerHTML = "FPS : " + fps;
    }

    private displayRendering(delay : number){
        this._renderingEl.innerHTML = "Rendering : " + delay + " ms";
    }

    private displayUpdate(delay : number){
        this._updateEl.innerHTML = "Update : " + delay + " ms";
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

}