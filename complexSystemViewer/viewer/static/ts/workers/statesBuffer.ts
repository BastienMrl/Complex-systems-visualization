import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";

export class StatesBuffer{

    private _states : Float32Array[];
    private _transformedValues : TransformableValues;

    private _isNewValues : boolean = false;
    private _valueIsReshaped : boolean = false;


    private _socketManager : SocketManager;

    


    constructor(){
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketManager = SocketManager.getInstance();
        this._socketManager.onDataReceived = function(data : any){
            this.onStateReceived(data);
        }.bind(this);
    };

    public get hasNewValue() : boolean {
        return this._isNewValues;
    }

    public get isReshaped() : boolean {
        return this._valueIsReshaped;
    }

    public get values() : TransformableValues{
        let values = this._transformedValues;
        this._transformedValues = TransformableValues.fromInstance(values);
        if (this._isNewValues){
            this._isNewValues = false;
            this._valueIsReshaped = false;
            this.requestState();
        }
        return values;
    }


    public flush(){
        this._isNewValues = false;
    }


    public requestState(){
        this._socketManager.requestData();
    }

    public onStateReceived(data : Array<Float32Array>){
        this._states = data;
        let nbElements = this.getNbElementsFromData(data);
        let nbChannels = this.getNbChannelsFromData(data);
        if (nbElements != this._transformedValues.nbElements || nbChannels != this._transformedValues.nbChannels){
            this._valueIsReshaped = true;
            this._transformedValues.reshape(nbElements, nbChannels);
        }
        
        this.transformState();
        this._isNewValues = true;
    }

    private getNbElementsFromData(data : any){
        return data[0][0];
    }

    private getNbChannelsFromData(data : any){
        return data[0][1];
    }


    public transformState(){
        this._transformedValues.setWithBackendValues(this._states);
    }
}