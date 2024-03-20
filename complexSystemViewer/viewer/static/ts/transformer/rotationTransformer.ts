import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";

export class RotationTransformer extends Transformer {
    private _min : number;
    private _max : number;

    public constructor (idx : number, inputVariable : string, channel : 0 | 1 | 2, min : number, max : number){
        super(idx, inputVariable);
        switch (channel){
            case 0:
                this.type = TransformType.ROTATION_X;
                break;
            case 1:
                this.type = TransformType.ROTATION_Y;
                break;
            case 2:
                this.type = TransformType.ROTATION_Z;
                break;
        }
        this._min = this.degToRad(min);
        this._max = this.degToRad(max);
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
            this._min = this.degToRad(params[0]);
        }
        if (params[1] != null){
            this._max = this.degToRad(params[1]);
        }
    }

    private degToRad(value : number) : number{
        return value * Math.PI / 180;
    }
}