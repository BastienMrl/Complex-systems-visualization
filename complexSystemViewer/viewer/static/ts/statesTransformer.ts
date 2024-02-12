import { Vec3 } from "./ext/glMatrix/index.js";

const sizePerColor = 3;
const sizePerTranslation = 3;

export enum TransformType {
    COLOR,
    COLOR_R,
    COLOR_G,
    COLOR_B,

    POSITION_X,
    POSITION_Y,
    POSITION_Z
}

export class TransformableValues{

    public colors : Float32Array;
    public translations : Float32Array;

    
    public constructor(nbElements : number){
        this.colors = new Float32Array(nbElements * sizePerColor).fill(0);
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0);
    }
}   


export class StatesTransformer{

    private _transformers : Transformer[];
    private _dataIndices : number[];

    public constructor(){
        this._transformers = [];
        this._dataIndices = [];
    }

    public addTransformer(type : TransformType, dataIndex : number, ...args : any[]) : number{
        switch(type){
            case TransformType.COLOR:
                this._transformers.push(new ColorTransformer(args[0], args[1]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.COLOR_R:
                break;
            case TransformType.COLOR_G:
                break;
            case TransformType.COLOR_B:
                break;
            
            case TransformType.POSITION_X:
                this._transformers.push(new PositionTransformer(0, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.POSITION_Y:
                this._transformers.push(new PositionTransformer(1, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.POSITION_Z:
                this._transformers.push(new PositionTransformer(2, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
        }

        return this._transformers.length - 1;
    }

    public applyTransformers(data : Float32Array[], values : TransformableValues){
        this._transformers.forEach((transformer, idx) => {
            transformer.transform(data[this._dataIndices[idx]], values);
        });
    }

    public setParams(id : number, ...args : any[]){
        if (id < 0 || id >= this._transformers.length)
            return;
        this._transformers[id].setParameters(args);
    }


}

abstract class Transformer {
    public type : TransformType;
    

    public abstract transform(states : Float32Array, values : TransformableValues) : void;
    public abstract setParameters(...args : any[]) : void;
}

class ColorTransformer extends Transformer{
    public type  : TransformType = TransformType.COLOR;

    private _colorMin : Vec3;
    private _colorMax : Vec3;

    public constructor(colorMin : number[], colorMax : number[]){
        super();
        this._colorMin = Vec3.fromValues(colorMin[0], colorMin[1], colorMin[2]);
        this._colorMax = Vec3.fromValues(colorMax[0], colorMax[1], colorMax[2]);
    }
    
    public transform(states: Float32Array, values : TransformableValues): void {
        states.forEach((alpha, idx) =>{
            let color = new Vec3().copy(this._colorMin).scale(1 - alpha);
            color.add(new Vec3().copy(this._colorMax).scale(alpha));
            for (let i = 0; i < 3; ++i)
                values.colors[idx * 3 + i] = color[i];
        });
    }

    public setParameters(...args: any[]): void {
        this._colorMin = Vec3.fromValues(args[0][0], args[0][1], args[0][2]);
        this._colorMax = Vec3.fromValues(args[1][0], args[1][1], args[1][2]);
    }
}

class PositionTransformer extends Transformer{
    private _axeIdx : number;

    private _factor : number

    
    public constructor(idx : number, factor : number = 1.){
        super();
        this._axeIdx = idx;
        this._factor = factor;
        switch(idx){
            case 0 : 
                this.type = TransformType.POSITION_X;
                break;
            case 1 :
                this.type = TransformType.POSITION_Y;
                break;
            case 2 : 
                this.type = TransformType.POSITION_Z;
                break;
            }
    }

    public transform(states: Float32Array, values: TransformableValues): void {
        states.forEach((value, idx) => {
            let outIndex = idx * 3 + this._axeIdx;
            values.translations[outIndex] = value * this._factor;
        });  
    }

    public setParameters(...args: any[]): void {
        this._factor = args[0];
    }
}