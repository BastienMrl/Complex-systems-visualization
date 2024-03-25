export class WorkerTimers {
    _transformationTimer;
    _parsingTimer;
    _receivingTimer;
    _transformationSample;
    _parsingSample;
    _receivingSample;
    static _instance;
    constructor() {
        //
    }
    static getInstance() {
        if (!WorkerTimers._instance)
            WorkerTimers._instance = new WorkerTimers();
        return WorkerTimers._instance;
    }
    startTransformationTimer() {
        this._transformationSample = performance.now();
    }
    stopTransformationTimer() {
        this._transformationTimer = performance.now() - this._transformationSample;
    }
    startParsingTimer() {
        this._parsingSample = performance.now();
    }
    stopParsingTimer() {
        this._parsingTimer = performance.now() - this._parsingSample;
    }
    startReceivingTimer() {
        this._receivingSample = performance.now();
    }
    stopReceivingTimer() {
        this._receivingTimer = performance.now() - this._receivingSample;
    }
    get transformationTimer() {
        return this._transformationTimer;
    }
    get parsingTimer() {
        return this._parsingTimer;
    }
    get receivingTimer() {
        return this._receivingTimer;
    }
}
