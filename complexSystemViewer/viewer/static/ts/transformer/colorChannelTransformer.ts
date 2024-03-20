import { ShaderFunction } from "../shaderUtils.js";
import * as Utils from "../typeUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";

export class ColorChannelTransformer extends Transformer {
    private _min : number;
    private _max : number;

    public constructor (idx : number, inputVariable : string, channel : 0 | 1 | 2, min : number = 0, max : number = 1){
        super(idx, inputVariable);
        this._min = Utils.mapValue(0, 255, 0, 1, min);
        this._max = Utils.mapValue(0, 255, 0, 1, max);
        switch(channel){
            case 0:
                this.type = TransformType.COLOR_R;
                break;
            case 1:
                this.type = TransformType.COLOR_G;
                break;
            case 2:
                this.type = TransformType.COLOR_B;
                break;
        }
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

    public setParameters(params : any[]) : void{
        if (params[0] != null)
            this._min = Utils.mapValue(0, 255, 0, 1, params[0]);
        if (params[1] != null)
            this._max = Utils.mapValue(0, 255, 0, 1, params[1]);
    }
}