import { TransformableValues } from "./statesTransformer.js";
import { SocketHandler } from "./socketHandler.js";
export class StatesBuffer {
    _states;
    _transformedValues;
    _socketHandler;
    _isInitialized;
    transformer;
    constructor(transformer) {
        this._states = [];
        this._transformedValues = new TransformableValues();
        this.transformer = transformer;
        this._socketHandler = SocketHandler.getInstance();
        this._isInitialized = false;
    }
    ;
    get isReady() {
        return this._isInitialized;
    }
    get values() {
        let values = this._transformedValues;
        this.requestState();
        return values;
    }
    initializeElements(nbElements) {
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketHandler.requestEmptyInstance(nbElements);
    }
    // TODO : use this request instead of requestRandomState, when transmission is operational
    requestState() {
        this._socketHandler.requestData();
    }
    onStateReceived(data) {
        this._states = data;
        this.transformState();
        this._isInitialized = true;
    }
    transformState() {
        // this.transformer.applyTransformers(this._states.shift(), this._transformedValues);
        this._transformedValues.states = new Float32Array(this._states[2]);
        // this._transformedValues.translations = new Float32Array(result);
        this._states[0].forEach((e, i) => {
            this._transformedValues.translations[i * 3] = e;
        });
        this._states[1].forEach((e, i) => {
            this._transformedValues.translations[i * 3 + 1] = e;
        });
    }
}
