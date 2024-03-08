import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";

export class StatesBuffer{

    private _states : Float32Array[];
    private _transformedValues : TransformableValues;

    private _isNewValues : boolean = false;


    private _socketManager : SocketManager;
    private _isInitialized : boolean;

    


    constructor(){
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketManager = SocketManager.getInstance();
        this._socketManager.onDataReceived = function(data : any){
            console.log("LOOOOOOOOOODGF")
            this.onStateReceived(data);
        }.bind(this);
        this._isInitialized = false;
    };

    public get isReady() : boolean{
        return this._isInitialized;
    }

    public get values() : TransformableValues{
        let values = this._transformedValues;
        this._transformedValues = TransformableValues.fromInstance(values);
        if (this._isNewValues){
            this._isNewValues = false;
            this.requestState();
        }
        return values;
    }

    public get hasNewValue() : boolean {
        return this._isNewValues;
    }

    public flush(){
        this._isNewValues = false;
    }

    

    public initializeElements(nbElements: number){
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketManager.resetSimulation(nbElements);
    }


    public requestState(){
        this._socketManager.requestData();
    }

    public onStateReceived(data : any){
        this._states = data;
        this.transformState();
        this._isInitialized = true;
        this._isNewValues = true;
    }


    public transformState(){
        this._transformedValues.setWithBackendValues(this._states);
    }
}