export class Stats{

    #fpsEl;
    #updateEl;
    #renderingEl;

    #renderingTimer;
    #updateTimer;

    #nbUpdates = 0
    #nbRendering = 0

    #fpsAccumulator = 0;
    #renderingAccumulator = 0;
    #updateAccumulator = 0;

    nbIteration = 10;

    constructor(fpsEl, updateEl, renderingEl){
        this.#fpsEl = fpsEl;
        this.#updateEl = updateEl;
        this.#renderingEl = renderingEl;
    }

    startRenderingTimer(delta){
        this.#renderingTimer = performance.now();
        this.#fpsAccumulator += delta;
        this.#nbRendering += 1;
        if (this.#nbRendering == this.nbIteration){
            let delay = this.#fpsAccumulator / this.nbIteration;
            this.#fpsAccumulator = 0;
            this.#displayFPS(Math.round(1. / delay));
        }
    }
    
    stopRenderingTimer(){
        let delta = performance.now() - this.#renderingTimer
        this.#renderingAccumulator += delta
        if (this.#nbRendering == this.nbIteration){
            let delay = this.#renderingAccumulator / this.nbIteration;
            this.#nbRendering = 0;
            this.#renderingAccumulator = 0;
            this.#displayRendering(delay);
        }
    }

    startUpdateTimer(){
        this.#updateTimer = performance.now();
    }

    stopUpdateTimer(){
        let delta = performance.now() - this.#updateTimer
        this.#updateAccumulator += delta;
        this.#nbUpdates += 1;
        if (this.#nbUpdates == this.nbIteration){
            let delay = this.#updateAccumulator / this.nbIteration;
            this.#nbUpdates = 0;
            this.#updateAccumulator = 0;
            this.#displayUpdate(delay);
        }
    }

    #displayFPS(fps){
        this.#fpsEl.innerHTML = "FPS : " + fps;
    }

    #displayRendering(delay){
        this.#renderingEl.innerHTML = "Rendering : " + delay + " ms";
    }

    #displayUpdate(delay){
        this.#updateEl.innerHTML = "Update : " + delay + " ms";
    }
}