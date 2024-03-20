import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";

export class ScalingTransformer extends Transformer {
    private _min : number;
    private _max : number;

    public type : TransformType = TransformType.SCALING;

    public constructor (idx : number, inputVariable : string, min : number, max : number){
        super(idx, inputVariable);
        this._min = min;
        this._max = max;
    }

    public applyTransformation(input: number): number | [number, number, number] {
        return this._min * (1 - input) + this._max * input;
    }

    public getParamsDeclarationBlock(): string {
        let s : string = "";
        s += this.getParamDeclaration(0, this._min) + "\n";
        s += this.getParamDeclaration(1, this._max);
        return s;
    }

    public getTransformationsBlock(): string {
        return this.getTransformerFunctionCall(ShaderFunction.INTERPOLATION, [0, 1]);    
    }

    public setParameters(params: any[]): void {
        if (params[0] != null){
            this._min = params[0];
        }
        if (params[1] != null){
            this._max = params[1];
        }
    }
}