import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { ShaderVariable } from "../shaderUtils.js";

type ShaderVariableType = number
                     | [number, number]
                     | [number, number, number]
                     | [number, number, number, number]
;

export abstract class Transformer {
    public type : TransformType;

    protected _id : number;
    protected _inputVariable : string;

    public constructor(id : number, inputVariable : string){
        this._id = id;
        this._inputVariable = inputVariable;
    }


    public abstract getParamsDeclarationBlock() : string;
    public abstract getTransformationsBlock() : string; 
    
    public abstract applyTransformation(input : number) : number | [number, number, number];

    private getTypeNbElements(value : ShaderVariableType) : 1 | 2 | 3 | 4 {
        if (Array.isArray(value))
            return value.length;
        return 1;
    }

    protected getOutputName() : string{
        switch (this.type){
            case TransformType.COLOR:
                return `${ShaderVariable.COLOR}`;
            case TransformType.COLOR_R:
                return `${ShaderVariable.COLOR}.r`;
            case TransformType.COLOR_G:
                return `${ShaderVariable.COLOR}.g`;
            case TransformType.COLOR_B:
                return `${ShaderVariable.COLOR}.b`;
            case TransformType.POSITION_X:
                return `${ShaderVariable.TRANSLATION}.x`;
            case TransformType.POSITION_Y:
                return `${ShaderVariable.TRANSLATION}.y`;
            case TransformType.POSITION_Z:
                return `${ShaderVariable.TRANSLATION}.z`;
            case TransformType.ROTATION_X:
                return `${ShaderVariable.ROTATION}.x`
            case TransformType.ROTATION_Y:
                return `${ShaderVariable.ROTATION}.y`
            case TransformType.ROTATION_Z:
                return `${ShaderVariable.ROTATION}.z`
            case TransformType.SCALING:
                return `${ShaderVariable.SCALING}`;
        }
    }

    private getTypeDeclaration(value : ShaderVariableType) : string{
        
        switch(this.getTypeNbElements(value)){
            case 1 :
                return "float";
            case 2 :
                return "vec2";
            case 3 :
                return "vec3";
            case 4 :
                return "vec4";
        }
    }

    private getVariableInitialisation(value : ShaderVariableType){
        let toFloatString = function(value : number){
            if (!`${value}`.includes(".")){
                let toFloat = parseFloat(`${value}`).toFixed(2);
                return toFloat;
            }
            return value;
        }
        
        switch(this.getTypeNbElements(value)){
            case 1:
                return `${toFloatString(value as number)}`;
            case 2:
                return `vec2(${toFloatString(value[0])}, ${toFloatString(value[1])})`;
            case 3:
                return `vec3(${toFloatString(value[0])}, ${toFloatString(value[1])}, ${toFloatString(value[2])})`;
            case 4:
                return `vec4(${toFloatString(value[0])}, ${toFloatString(value[1])}, ${toFloatString(value[2])}, ${toFloatString(value[3])})`;
        }
    }

    protected getParamName(paramIdx : number) : string{
        return `param_${this._id}_${paramIdx}`;
    }

    protected getParamDeclaration(paramIdx : number, value : ShaderVariableType){
        return `const ${this.getTypeDeclaration(value)} param_${this._id}_${paramIdx} = ${this.getVariableInitialisation(value)};`;
    }

    protected getTransformerFunctionCall(fct : ShaderFunction, paramsIdx : number[]){
        let s = `${fct}(${this.getOutputName()}`;
        paramsIdx.forEach(e => {
            s += `, ${this.getParamName(e)}`;
        });
        s += `, ${this._inputVariable});`;
        return s;
    }
    

    public abstract setParameters(params : any[]) : void;

    public setInputVariable(variable : string){
        this._inputVariable = variable;
    }

    public getInputVariable() : string{
        return this._inputVariable;
    }

    public getId() : number{
        return this._id;
    }

}