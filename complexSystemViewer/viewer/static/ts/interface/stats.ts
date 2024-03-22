export class Stats{
    private static _nbDigits = 1;
    private static readonly perfMessage = "PERF";
    private static readonly shapeMessage = "SHAPE";
    private static readonly modelMessage = "MODEL";

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


    private _totalEl : HTMLElement;

    private _nbIteration : number = 10;

    private _withLog : boolean = false;

    public constructor(fpsEl : HTMLElement, updateEl : HTMLElement, renderingEl : HTMLElement, pickingEl : HTMLElement, totalEl : HTMLElement){
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
        this._pickingEl = pickingEl;
        this._totalEl = totalEl;
    }

    public set withLog(value : boolean){
        this._withLog = value
    }

    public get withLog() : boolean {
        return this._withLog;
    }
    
    private displayFPS(fps : number){
        this._fpsEl.innerHTML = "FPS : " + fps.toFixed(0);
        const total =  this._currentRenderingDelay + this._currentUpdateDelay;
        this._totalEl.innerHTML = "Total : " + total.toFixed(Stats._nbDigits) + " ms";
        this.logPerformance("fps", fps);
    }

    private displayRendering(delay : number){
        this._renderingEl.innerHTML = "Rendering : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentRenderingDelay = delay;
        this.logPerformance("rendering", delay);
    }

    private displayUpdate(delay : number){
        this._updateEl.innerHTML = "Update : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentUpdateDelay = delay;
        this.logPerformance("updating", delay);
    }

    private displayPicking(delay : number){
        this._pickingEl.innerHTML = "Picking : " + delay.toFixed(Stats._nbDigits) + " ms";
        this.logPerformance("picking", delay)
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
        this.displayPicking(delta);
    }

    private logPerformance(name : string, value : number){
        if (!this.withLog) return;
        let s = `${Stats.perfMessage}/${name}/${value}`;
        console.info(s);
    }

    public logShape(nbElement : number, nbChannel : number){
        if (!this.withLog) return;
        let s = `${Stats.shapeMessage}/${nbElement}/${nbChannel}`;
        console.info(s);
    }

    public logModel(model : string){
        if (!this.withLog) return;
        let s = `${Stats.modelMessage}/${model}`;
        console.info(s);
    }
}