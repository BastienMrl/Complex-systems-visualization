import { TransformableValues } from "../transformableValues.js";
import { SocketHandler } from "./socketHandler.js";
export class StatesBuffer {
    _states;
    _transformedValues;
    _socketHandler;
    _isInitialized;
    constructor() {
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketHandler = SocketHandler.getInstance();
        this._socketHandler.onDataReceived = function (data) {
            
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
        this._socketHandler.stop();
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketHandler.requestEmptyInstance(nbElements);
        this._socketHandler.start(nbElements);
    }
    requestState() {
        this._socketHandler.requestData();
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
