import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";

export class PositionTransformer extends Transformer{
    private _factor : number;

    public constructor(idx : number, inputVariable : string, axis : 0 | 1 | 2, factor : number = 1.){
        super(idx, inputVariable);
        this.setFactor(factor);
        switch(axis){
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

    public applyTransformation(input: number): number | [number, number, number] {
        return input * this.factor;
    }

    private setFactor(factor : number){
        this._factor = factor;
    }

    public getParamsDeclarationBlock(): string {
        return this.getParamDeclaration(0, this._factor);
    }

    public getTransformationsBlock(): string {
        return this.getTransformerFunctionCall(ShaderFunction.FACTOR, [0]);    
    }


    public setParameters(params : any[]): void {
        this.setFactor(params[0]);
    }

    public get factor() : number{
        return this._factor;
    }
}