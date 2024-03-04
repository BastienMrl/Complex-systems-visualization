import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
export class StatesBuffer {
    _states;
    _transformedValues;
    _socketManager;
    _isInitialized;
    constructor() {
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketManager = SocketManager.getInstance();
        this._socketManager.onDataReceived = function (data) {
            this.onStateReceived(data);
        }.bind(this);
        this._isInitialized = false;
    }
    ;
    get isReady() {
        return this._isInitialized;
    }
    get values() {
        let values = this._transformedValues;
        this._transformedValues = TransformableValues.fromInstance(values);
        this.requestState();
        return values;
    }
    initializeElements(nbElements) {
        this._socketManager.stop();
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketManager.requestEmptyInstance(nbElements);
        this._socketManager.start(nbElements);
    }
    requestState() {
        this._socketManager.requestData();
    }
    onStateReceived(data) {
        this._states = data;
        this.transformState();
        this._isInitialized = true;
    }
    transformState() {
        this._transformedValues.states = new Float32Array(this._states[2]);
        this._states[0].forEach((e, i) => {
            this._transformedValues.translations[i * 3] = e;
        });
        this._states[1].forEach((e, i) => {
            this._transformedValues.translations[i * 3 + 1] = e;
        });
    }
}
