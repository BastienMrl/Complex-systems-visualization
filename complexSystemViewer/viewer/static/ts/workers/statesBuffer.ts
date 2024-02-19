import { TransformableValues } from "../transformableValues.js";
import { SocketHandler } from "./socketHandler.js";


export class StatesBuffer{

    private _states : Float32Array[];
    private _transformedValues : TransformableValues;


    private _socketHandler : SocketHandler;
    private _isInitialized : boolean;

    


    constructor(){
        this._states = [];
        this._transformedValues = new TransformableValues();
        this._socketHandler = SocketHandler.getInstance();
        this._socketHandler.onDataReceived = function(data : any){
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
        this.requestState();
        return values;
    }

    

    public initializeElements(nbElements: number){
        this._socketHandler.stop();
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketHandler.requestEmptyInstance(nbElements);
        this._socketHandler.start(nbElements);
    }


    public requestState(){
        this._socketHandler.requestData();
    }

    public onStateReceived(data : any){
        this._states = data;
        this.transformState();
        this._isInitialized = true;
    }


    public transformState(){
        this._transformedValues.states = new Float32Array(this._states[2])

        this._states[0].forEach((e, i) =>{
            this._transformedValues.translations[i * 3] = e;
        });
        this._states[1].forEach((e, i) =>{
            this._transformedValues.translations[i * 3 + 1] = e;
        })
    }
}