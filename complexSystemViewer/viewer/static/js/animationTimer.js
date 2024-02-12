export class AnimationTimer {
    _duration;
    _startingTime;
    loop;
    _isRunning;
    _callback;
    _defaultInterpolationCurve;
    _interpolationCurves;
    constructor(duration, loop) {
        this._duration = duration * 1000;
        this.loop = loop;
        this._defaultInterpolationCurve = function (time) {
            // linear
            return time;
        };
        this._isRunning = false;
        this._interpolationCurves = [];
        this._callback = function () { };
    }
    set callback(callback) {
        this._callback = callback;
    }
    set duration(duration) {
        this._duration = duration;
    }
    onTimeout() {
        if (!this._isRunning)
            return;
        this._isRunning = false;
        this._callback();
        if (this.loop)
            this.play();
    }
    play() {
        this._startingTime = performance.now();
        this._isRunning = true;
        setTimeout(this.onTimeout.bind(this), this._duration);
    }
    stop() {
        this._isRunning = false;
    }
    // returns id
    addAnimationCurve(fct) {
        return this._interpolationCurves.push(fct) - 1;
    }
    getAnimationTime(idx) {
        if (!this._isRunning)
            return 0.;
        let time = performance.now() - this._startingTime;
        time /= this._duration;
        time = time > 1. ? 1. : time;
        if (idx == undefined || idx < 0 || idx >= this._interpolationCurves.length)
            return this._defaultInterpolationCurve(time);
        return this._interpolationCurves[idx](time);
    }
}
