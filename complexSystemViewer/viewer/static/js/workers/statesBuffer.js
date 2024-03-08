import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
export class StatesBuffer {
    _states;
    _transformedValues;
    _isNewValues = false;
    _socketManager;
    _isInitialized;
    constructor() {
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketManager = SocketManager.getInstance();
        this._socketManager.onDataReceived = function (data) {
            console.log("LOOOOOOOOOODGF");
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
        if (this._isNewValues) {
            this._isNewValues = false;
            this.requestState();
        }
        return values;
    }
    get hasNewValue() {
        return this._isNewValues;
    }
    flush() {
        this._isNewValues = false;
    }
    initializeElements(nbElements) {
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketManager.resetSimulation(nbElements);
    }
    requestState() {
        this._socketManager.requestData();
    }
    onStateReceived(data) {
        this._states = data;
        this.transformState();
        this._isInitialized = true;
        this._isNewValues = true;
    }
    transformState() {
        this._transformedValues.setWithBackendValues(this._states);
    }
}
