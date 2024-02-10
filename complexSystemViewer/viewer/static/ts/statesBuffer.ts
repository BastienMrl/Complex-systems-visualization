import { Vec3 } from "./glMatrix/index.js";
import { StatesTransformer, TransformableValues } from "./statesTransformer.js";



export class StatesBuffer{

    private _states : Float32Array[][];

    private _transformedValues : TransformableValues;
    
    private _nbElements : number;

    public transformer : StatesTransformer;
    
    constructor(nbElements : number, transformer : StatesTransformer){
        this._states = [];
        this._nbElements = nbElements
        this._transformedValues = new TransformableValues(this._nbElements);
        this.transformer = transformer;
        this.requestRandomState();
    };

    public get values() : TransformableValues{
        let values = this._transformedValues;
        this.transformState();
        return values;
    }

    // TODO : use this request instead of requestRandomState, when transmission is operational
    public requestState(){
        //
    }

    public onStateReceived(data : any){
        // TODO : push data in _states, launch transformState
    }

    public requestRandomState(){
        const offset = 2.05;
        const sqrtInstances = Math.sqrt(this._nbElements);
        const nbRow = sqrtInstances;
        const nbCol = nbRow;
        const offsetRow = Vec3.fromValues(0, 0, offset);
        const offsetCol = Vec3.fromValues(offset, 0, 0);
        const center = -(nbRow - 1) * offset / 2.;
        const firstPos = Vec3.fromValues(center, 0, center);

        let x = new Float32Array(this._nbElements);
        let y = new Float32Array(this._nbElements);
        let state = new Float32Array(this._nbElements);
        
        let rowPos = firstPos;
        for (let i = 0; i < nbRow; i++) {
            let colPos = new Vec3().copy(rowPos);
            for (let j = 0; j < nbCol; j++) {
                let index = nbCol * i + j;
                x[index] = colPos[0];
                y[index] = colPos[2];
                state[index] = Math.round(Math.random());
                colPos.add(offsetCol);
            }
            rowPos.add(offsetRow);
        }

        this._states.push([x, y, state]);
    }

    public transformState(){
        this.transformer.applyTransformers(this._states.shift(), this._transformedValues);
        this.requestRandomState();
    }
}