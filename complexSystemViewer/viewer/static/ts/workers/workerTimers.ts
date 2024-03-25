export class WorkerTimers{
    private _transformationTimer : number;
    private _parsingTimer : number;
    private _receivingTimer : number;

    private _transformationSample : number;
    private _parsingSample : number;
    private _receivingSample : number;

    private static _instance : WorkerTimers

    private constructor(){
        //
    }
    public static getInstance() : WorkerTimers{
        if (!WorkerTimers._instance)
            WorkerTimers._instance = new WorkerTimers();
        return WorkerTimers._instance;
    }

    public startTransformationTimer(){
        this._transformationSample = performance.now();
    }

    public stopTransformationTimer(){
        this._transformationTimer = performance.now() - this._transformationSample;
    }

    public startParsingTimer(){
        this._parsingSample = performance.now();
    }

    public stopParsingTimer(){
        this._parsingTimer = performance.now() - this._parsingSample;
    }

    public startReceivingTimer(){
        this._receivingSample = performance.now();
    }

    public stopReceivingTimer(){
        this._receivingTimer = performance.now() - this._receivingSample;
    }

    public get transformationTimer() : number{
        return this._transformationTimer;
    }

    public get parsingTimer() : number{
        return this._parsingTimer;
    }

    public get receivingTimer() : number{
        return this._receivingTimer;
    }

}