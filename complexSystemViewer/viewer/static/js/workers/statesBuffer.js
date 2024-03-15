import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
export class StatesBuffer {
    _states;
    _transformedValues;
    _isNewValues = false;
    _valueIsReshaped = false;
    _socketManager;
    constructor() {
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketManager = SocketManager.getInstance();
        this._socketManager.onDataReceived = function (data) {
            this.onStateReceived(data);
        }.bind(this);
    }
    ;
    get hasNewValue() {
        return this._isNewValues;
    }
    get isReshaped() {
        return this._valueIsReshaped;
    }
    get values() {
        let values = this._transformedValues;
        this._transformedValues = TransformableValues.fromInstance(values);
        if (this._isNewValues) {
            this._isNewValues = false;
            this._valueIsReshaped = false;
            this.requestState();
        }
        return values;
    }
    flush() {
        this._isNewValues = false;
    }
    requestState() {
        this._socketManager.requestData();
    }
    onStateReceived(data) {
        this._states = data;
        let nbElements = this.getNbElementsFromData(data);
        let nbChannels = this.getNbChannelsFromData(data);
        if (nbElements != this._transformedValues.nbElements || nbChannels != this._transformedValues.nbChannels) {
            this._valueIsReshaped = true;
            this._transformedValues.reshape(nbElements, nbChannels);
        }
        this.transformState();
        this._isNewValues = true;
    }
    getNbElementsFromData(data) {
        return data[0][0];
    }
    getNbChannelsFromData(data) {
        return data[0][1];
    }
    transformState() {
        this._transformedValues.setWithBackendValues(this._states);
    }
}
