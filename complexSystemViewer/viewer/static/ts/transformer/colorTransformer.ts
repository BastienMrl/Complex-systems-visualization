import { ShaderFunction } from "../shaderUtils.js";
import * as Utils from "../typeUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";

export class ColorTransformer extends Transformer{
    public type  : TransformType = TransformType.COLOR;

    private _colorMin : [number, number, number];
    private _colorMax : [number, number, number];

    public constructor(id : number, inputVariable : string, params : any[]){
        super(id, inputVariable);
        if (typeof params[0] == "string")
            this._colorMin = Utils.hexToRgbA(params[0]);
        else 
            this._colorMin = [0., 0., 0.];
        if (typeof params[1] == "string")
            this._colorMax = Utils.hexToRgbA(params[1]);
        else 
            this._colorMax = [1., 1., 1.];
    }

    public applyTransformation(input: number): number | [number, number, number] {
        let ret = this._colorMin.map(x => x * (1 - input))
        return ret.map((x, i) => x + this._colorMax[i] * input) as [number, number, number];
    }

    public getParamsDeclarationBlock(): string {        
        let s : string = "";
        s += this.getParamDeclaration(0, this._colorMin) + "\n";
        s += this.getParamDeclaration(1, this._colorMax);
        return s;
    }

    public getTransformationsBlock(): string {
        return this.getTransformerFunctionCall(ShaderFunction.INTERPOLATION, [0, 1]);
    }
    

    public setParameters(params : any[]): void {
        if (typeof params[0] == "string")
            this._colorMin = Utils.hexToRgbA(params[0]);
        if (typeof params[1] == "string")
            this._colorMax = Utils.hexToRgbA(params[1]);
    }
}