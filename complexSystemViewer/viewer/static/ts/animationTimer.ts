export class AnimationTimer{
    private _duration : number;
    private _startingTime : number;

    public loop : boolean;
    private _isRunning : boolean;

    private _callback : () => void;

    private _defaultInterpolationCurve : (time : number) => number;

    private _interpolationCurves : ((time : number) => number)[];


    
    constructor(duration : number, loop : boolean){
        this._duration = duration * 1000;
        this.loop = loop;
        this._defaultInterpolationCurve = function(time : number){
            // linear
            return time;
        };
        this._isRunning = false;
        this._interpolationCurves = []
        this._callback = function(){};
    }

    public set callback(callback : () => void){
        this._callback = callback;
    }

    public set duration(duration : number){
        this._duration = duration * 1000;
    }

    public get isRunning() : boolean{
        return this._isRunning;
    }

    private onTimeout(){
        if (!this._isRunning)
            return;
        this._isRunning = false;
        this._callback();
        if (this.loop)
            this.play();
    }

    public play(){
        if (this._isRunning)
            return;
        this._startingTime = performance.now();
        this._isRunning = true;
        setTimeout(this.onTimeout.bind(this), this._duration);
    }

    public stop(){
        this._isRunning = false;
    }

    // returns id
    public addAnimationCurve(fct : (time : number) => number) : number{
        return this._interpolationCurves.push(fct) - 1;
    }

    public getAnimationTime(idx? : number) : number {
        if (!this._isRunning) return 0.;

        let time = performance.now() - this._startingTime;
        time /= this._duration;
        time = time > 1. ? 1. : time;

        if (idx == undefined || idx < 0 || idx >= this._interpolationCurves.length)
            return this._defaultInterpolationCurve(time);
        return this._interpolationCurves[idx](time);
    }
}