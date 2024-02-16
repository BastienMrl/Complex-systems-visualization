import { Vec3 } from "./ext/glMatrix/index.js";
import { StatesTransformer, TransformableValues } from "./statesTransformer.js";
import { SocketHandler } from "./socketHandler.js";



export class StatesBuffer{

    private _states : Float32Array[];
    private _transformedValues : TransformableValues;
    private _socketHandler : SocketHandler;

    private _isInitialized : boolean;

    


    public transformer : StatesTransformer;


    constructor(transformer : StatesTransformer){
        this._states = [];
        this._transformedValues = new TransformableValues();
        this.transformer = transformer;
        this._socketHandler = SocketHandler.getInstance();
        this._isInitialized = false;
    };

    public get isReady() : boolean{
        return this._isInitialized;
    }

    public get values() : TransformableValues{
        let values = this._transformedValues;
        this.requestState();
        return values;
    }

    public initializeElements(nbElements: number){
        this._isInitialized = false;
        this._transformedValues.reshape(nbElements);
        this._socketHandler.requestEmptyInstance(nbElements);
    }



    // TODO : use this request instead of requestRandomState, when transmission is operational
    public requestState(){
        this._socketHandler.requestData();
    }

    public onStateReceived(data : any){
        this._states = data;
        this.transformState();
        this._isInitialized = true;
    }


    public transformState(){
        // this.transformer.applyTransformers(this._states.shift(), this._transformedValues);
        this._transformedValues.states = new Float32Array(this._states[2])

        // this._transformedValues.translations = new Float32Array(result);
        this._states[0].forEach((e, i) =>{
            this._transformedValues.translations[i * 3] = e;
        });
        this._states[1].forEach((e, i) =>{
            this._transformedValues.translations[i * 3 + 1] = e;
        })
    }
}