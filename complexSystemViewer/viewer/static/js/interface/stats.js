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
    _pickingEl;
    _pickingTimer;
    _transformationEl;
    _parsingEl;
    _receivingEl;
    _totalEl;
    _nbIteration = 10;
    _withLog = false;
    constructor(fpsEl, updateEl, renderingEl, pickingEl, totalEl, transformationEl, parsingEl, receivingEl) {
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
        this._pickingEl = pickingEl;
        this._totalEl = totalEl;
        this._transformationEl = transformationEl;
        this._parsingEl = parsingEl;
        this._receivingEl = receivingEl;
    }
    set withLog(value) {
        this._withLog = value;
    }
    get withLog() {
        return this._withLog;
    }
    displayFPS(fps) {
        this._fpsEl.innerHTML = "FPS : " + fps.toFixed(0);
        const total = this._currentRenderingDelay + this._updateTimer;
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
        this._updateTimer = delay;
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
        this.displayUpdate(delta);
    }
    startPickingTimer() {
        this._pickingTimer = performance.now();
    }
    stopPickingTimer() {
        const delta = performance.now() - this._pickingTimer;
        this.displayPicking(delta);
    }
    displayWorkerTimer(name, value) {
        switch (name) {
            case ("transformation"):
                this._transformationEl.innerText = `${value}`;
                this.logPerformance(name, value);
                break;
            case ("parsing"):
                this._parsingEl.innerText = `${value}`;
                this.logPerformance(name, value);
                break;
            case ("receiving"):
                this._receivingEl.innerText = `${value}`;
                this.logPerformance(name, value);
                break;
        }
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
