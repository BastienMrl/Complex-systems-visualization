export class Stats {
    _fpsEl;
    _updateEl;
    _renderingEl;
    _renderingTimer;
    _updateTimer;
    _nbUpdates = 0;
    _nbRendering = 0;
    _fpsAccumulator = 0;
    _renderingAccumulator = 0;
    _updateAccumulator = 0;
    _nbIteration = 10;
    constructor(fpsEl, updateEl, renderingEl) {
        this._fpsEl = fpsEl;
        this._updateEl = updateEl;
        this._renderingEl = renderingEl;
    }
    displayFPS(fps) {
        this._fpsEl.innerHTML = "FPS : " + fps;
    }
    displayRendering(delay) {
        this._renderingEl.innerHTML = "Rendering : " + delay + " ms";
    }
    displayUpdate(delay) {
        this._updateEl.innerHTML = "Update : " + delay + " ms";
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
}
