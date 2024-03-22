export class Stats {
    static _nbDigits = 1;
    static perfMessage = "PERF";
    static shapeMessage = "SHAPE";
    static modelMessage = "MODEL";
    _fpsEl;
    _fpsAccumulator = 0;
    _renderingEl;
    _renderingTimer;
    _nbRendering = 0;
    _renderingAccumulator = 0;
    _currentRenderingDelay = 0;
    _updateEl;
    _updateTimer;
    _nbUpdates = 0;
    _updateAccumulator = 0;
    _currentUpdateDelay = 0;
    _pickingEl;
    _pickingTimer;
    _totalEl;
    _nbIteration = 10;
    _withLog = false;
    constructor(fpsEl, updateEl, renderingEl, pickingEl, totalEl) {
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
        this._pickingEl = pickingEl;
        this._totalEl = totalEl;
    }
    set withLog(value) {
        this._withLog = value;
    }
    get withLog() {
        return this._withLog;
    }
    displayFPS(fps) {
        this._fpsEl.innerHTML = "FPS : " + fps.toFixed(0);
        const total = this._currentRenderingDelay + this._currentUpdateDelay;
        this._totalEl.innerHTML = "Total : " + total.toFixed(Stats._nbDigits) + " ms";
        this.logPerformance("fps", fps);
    }
    displayRendering(delay) {
        this._renderingEl.innerHTML = "Rendering : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentRenderingDelay = delay;
        this.logPerformance("rendering", delay);
    }
    displayUpdate(delay) {
        this._updateEl.innerHTML = "Update : " + delay.toFixed(Stats._nbDigits) + " ms";
        this._currentUpdateDelay = delay;
        this.logPerformance("updating", delay);
    }
    displayPicking(delay) {
        this._pickingEl.innerHTML = "Picking : " + delay.toFixed(Stats._nbDigits) + " ms";
        this.logPerformance("picking", delay);
    }
    startRenderingTimer(delta) {
        this._renderingTimer = performance.now();
        this._fpsAccumulator += delta;
        this._nbRendering += 1;
        if (this._nbRendering == this._nbIteration) {
            const delay = this._fpsAccumulator / this._nbIteration;
            this._fpsAccumulator = 0;
            this.displayFPS(Math.round(1. / delay));
        }
    }
    stopRenderingTimer() {
        const delta = performance.now() - this._renderingTimer;
        this._renderingAccumulator += delta;
        if (this._nbRendering == this._nbIteration) {
            const delay = this._renderingAccumulator / this._nbIteration;
            this._nbRendering = 0;
            this._renderingAccumulator = 0;
            this.displayRendering(delay);
        }
    }
    startUpdateTimer() {
        this._updateTimer = performance.now();
    }
    stopUpdateTimer() {
        const delta = performance.now() - this._updateTimer;
        this._updateAccumulator += delta;
        this._nbUpdates += 1;
        if (this._nbUpdates == this._nbIteration) {
            const delay = this._updateAccumulator / this._nbIteration;
            this._nbUpdates = 0;
            this._updateAccumulator = 0;
            this.displayUpdate(delay);
        }
    }
    startPickingTimer() {
        this._pickingTimer = performance.now();
    }
    stopPickingTimer() {
        const delta = performance.now() - this._pickingTimer;
        this.displayPicking(delta);
    }
    logPerformance(name, value) {
        if (!this.withLog)
            return;
        let s = `${Stats.perfMessage}/${name}/${value}`;
        console.info(s);
    }
    logShape(nbElement, nbChannel) {
        if (!this.withLog)
            return;
        let s = `${Stats.shapeMessage}/${nbElement}/${nbChannel}`;
        console.info(s);
    }
    logModel(model) {
        if (!this.withLog)
            return;
        let s = `${Stats.modelMessage}/${model}`;
        console.info(s);
    }
}
