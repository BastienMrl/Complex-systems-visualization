import { ShaderFunction, ShaderMeshInputs, ShaderUniforms } from "../shaderUtils.js";
import { InputType } from "./inputType.js";
import { Transformer } from "./transformer.js";
import { TransformType } from "./transformType.js";
import { ColorTransformer } from "./colorTransformer.js";
import { ColorChannelTransformer } from "./colorChannelTransformer.js";
import { ScalingTransformer } from "./scalingTransformer.js";
import { PositionTransformer } from "./positionTransformer.js";
import { RotationTransformer } from "./rotationTransformer.js";

export class TransformerBuilder{

    private _transformers : Transformer[];

    private _idCpt : number;
    private _inputDeclarations : string[];

    public constructor(){
        this._transformers = [];
        this._idCpt = 0;
        this._inputDeclarations = [];
    }

    private addInputVariableDeclaration(transformType : TransformType, intputType : InputType, name : string){
        let s = `float ${name} = `;
        let onT0 = "";
        let onT1 = "";
        let time = "";
        let normalized = false;
        let need_normalization = false;
        let normalization_axis = null;
        switch(transformType){
            case TransformType.COLOR:
            case TransformType.COLOR_R:
            case TransformType.COLOR_G:
            case TransformType.COLOR_B:
                time = ShaderUniforms.TIME_COLOR;
                need_normalization = true;
                break;
            case TransformType.POSITION_X:
            case TransformType.POSITION_Y:
            case TransformType.POSITION_Z:
                time = ShaderUniforms.TIME_TRANSLATION;
                break;
            case TransformType.SCALING:
                time = ShaderUniforms.TIME_SCALING
                break;
            case TransformType.ROTATION_X:
            case TransformType.ROTATION_Y:
            case TransformType.ROTATION_Z:
                time = ShaderUniforms.TIME_ROTATION;
                break;
            
        }
        switch(intputType){
            case InputType.POSITION_X:
                onT0 = ShaderMeshInputs.TRANSLATION_T0 + ".x";
                onT1 = ShaderMeshInputs.TRANLSATION_T1 + ".x";
                normalized = true && need_normalization;
                normalization_axis = 0;
                break;
            case InputType.POSITION_Y:
                onT0 = ShaderMeshInputs.TRANSLATION_T0 + ".y";
                onT1 = ShaderMeshInputs.TRANLSATION_T1 + ".y";
                normalized = true && need_normalization;
                normalization_axis = 2;
                break;
            case InputType.POSITION_Z:
                onT0 = ShaderMeshInputs.TRANSLATION_T0 + ".z";
                onT1 = ShaderMeshInputs.TRANLSATION_T1 + ".z";
                normalized = true && need_normalization;
                normalization_axis = 1;
                break;
            case InputType.STATE_0:
                onT0 = ShaderMeshInputs.STATE_0_T0;
                onT1 = ShaderMeshInputs.STATE_0_T1;
                break;
            case InputType.STATE_1:
                onT0 = ShaderMeshInputs.STATE_1_T0;
                onT1 = ShaderMeshInputs.STATE_1_T1;
                break;
            case InputType.STATE_2:
                onT0 = ShaderMeshInputs.STATE_2_T0;
                onT1 = ShaderMeshInputs.STATE_2_T1;
                break;
            case InputType.STATE_3:
                onT0 = ShaderMeshInputs.STATE_3_T0;
                onT1 = ShaderMeshInputs.STATE_3_T1;
                break;
        }
        s += `mix(${onT0}, ${onT1}, ${time});`;
        if (normalized)
            s += `\n${ShaderFunction.NORMALIZE_POSITION}(${name}, ${normalization_axis});`;
        this._inputDeclarations.push(s);
    }

    private deleteVariableDeclaration(variable : string){
        let idx = -1;
        for (let i = 0; i < this._inputDeclarations.length; ++i){
            if (this._inputDeclarations[i].includes(`${variable}`))
                idx = i;
        }
        if (idx >= 0)
            this._inputDeclarations.splice(idx, 1);
    }

    private getInputVariableName(transformType : TransformType, intputType : InputType) : string{
        let s = "input_";
        switch(transformType){
            case TransformType.COLOR:
            case TransformType.COLOR_R:
            case TransformType.COLOR_G:
            case TransformType.COLOR_B:
                s += "c";
                break;
            case TransformType.POSITION_X:
            case TransformType.POSITION_Y:
            case TransformType.POSITION_Z:
                s += "t";
                break;
            case TransformType.SCALING:
                s += "s";
                break;
            case TransformType.ROTATION_X:
            case TransformType.ROTATION_Y:
            case TransformType.ROTATION_Z:
                s += "r";
                break;
                
        }
        s += "_";
        switch(intputType){
            case InputType.POSITION_X:
                s += "x";
                break;
            case InputType.POSITION_Y:
                s += "y";
                break;
            case InputType.POSITION_Z:
                s += "z";
                break;
            case InputType.STATE_0:
                s += "s_0";
                break;
            case InputType.STATE_1:
                s += "s_1";
                break;
            case InputType.STATE_2:
                s += "s_2";
                break;
            case InputType.STATE_3:
                s += "s_3";
                break;
        }
        return s;
    }

    public addTransformer(type : TransformType, inputType : InputType, params? : any[]) : number{        
        let inputVariable = this.getInputVariableName(type, inputType);
        let id = this._idCpt++;
        switch(type){
            case TransformType.COLOR:
                this._transformers.push(new ColorTransformer(id, inputVariable, params));
                break;
            case TransformType.COLOR_R:
                this._transformers.push(new ColorChannelTransformer(id, inputVariable, 0, params[0], params[1]));
                break;
            case TransformType.COLOR_G:
                this._transformers.push(new ColorChannelTransformer(id, inputVariable, 1, params[0], params[1]));
                break;
            case TransformType.COLOR_B:
                this._transformers.push(new ColorChannelTransformer(id, inputVariable, 2, params[0], params[1]));
                break;
            case TransformType.POSITION_X:
                this._transformers.push(new PositionTransformer(id, inputVariable, 0, params == undefined ? 1. : params[0]));
                break;
            case TransformType.POSITION_Y:
                this._transformers.push(new PositionTransformer(id, inputVariable, 1, params == undefined ? 1. : params[0]));
                break;
            case TransformType.POSITION_Z:
                this._transformers.push(new PositionTransformer(id, inputVariable, 2, params == undefined ? 1. : params[0]));
                break;
            case TransformType.SCALING:
                this._transformers.push(new ScalingTransformer(id, inputVariable, params[0], params[1]));
                break;

            case TransformType.ROTATION_X:
                this._transformers.push(new RotationTransformer(id, inputVariable, 0, params[0], params[1]));
                break;
            case TransformType.ROTATION_Y:
                this._transformers.push(new RotationTransformer(id, inputVariable, 1, params[0], params[1]));
                break;
            case TransformType.ROTATION_Z:
                this._transformers.push(new RotationTransformer(id, inputVariable, 2, params[0], params[1]));
                break;
        }
        this.addInputVariableDeclaration(type, inputType, inputVariable);
        return id;
    }

    public removeTransformer(id : number){
        let transformer = this.getTransformerFromId(id);
        if(transformer == null)
            return;
        let variable = transformer.getInputVariable();
        this.deleteVariableDeclaration(variable);
        this._transformers.splice(this._transformers.indexOf(transformer), 1);
    }

    public generateTransformersBlock(){
        let inputDeclarations = "";
        let uniques = this._inputDeclarations.filter((value, index, array) => array.indexOf(value) === index);
        uniques.forEach((e) => {
            inputDeclarations += e + "\n";
        });
        
        let constants = "";
        let fctCalls = "";
        this._transformers.forEach((transformer) => {
            constants += transformer.getParamsDeclarationBlock() + "\n";
            fctCalls += transformer.getTransformationsBlock() + "\n";
        });
        return `${inputDeclarations}\n${constants}\n${fctCalls}`;
    }

    public generateTranslationTransformersBlock(){
        let inputDeclarations = "";
        let uniques = this._inputDeclarations.filter((value, index, array) => array.indexOf(value) === index);
        uniques.forEach((e) => {
            if (e.includes("_t_"))
                inputDeclarations += e + "\n";
        });
        let constants = "";
        let fctCalls = "";
        this._transformers.forEach((transformer) => {
            const t = transformer.type;
            if (t == TransformType.POSITION_X ||
                t == TransformType.POSITION_Y ||
                t == TransformType.POSITION_Z){

                constants += transformer.getParamsDeclarationBlock() + "\n";
                fctCalls += transformer.getTransformationsBlock() + "\n";
            }
        });
        return `${inputDeclarations}\n${constants}\n${fctCalls}`;
    }

    public setParams(id : number, params : any[]){
        let transformer : Transformer = this.getTransformerFromId(id);
        if (transformer == null)
            return;
        transformer.setParameters(params);
    }

    public setInputType(id : number, inputType : InputType){
        let transformer : Transformer = this.getTransformerFromId(id);
        if (transformer == null)
            return;

        let oldVariable = transformer.getInputVariable();
        let transformType = transformer.type;
        let newVariable = this.getInputVariableName(transformType, inputType);

        this.addInputVariableDeclaration(transformType, inputType, newVariable);
        transformer.setInputVariable(newVariable);
        
        this.deleteVariableDeclaration(oldVariable);
    }

    public getPositionFactor(axis : 0 | 1 | 2) : number{
        let type = TransformType.POSITION_X;
        switch(axis){
            case 0:
                type = TransformType.POSITION_X;
                break;
            case 1:
                type = TransformType.POSITION_Y;
                break;
            case 2:
                type = TransformType.POSITION_Z;
                break;
        }
        let factor = 0;
        this._transformers.forEach((e) => {
            if (e.type == type)
                factor += (e as PositionTransformer).factor;
        })
        return factor;
    }

    public getTransformerFromId(id : number) : Transformer{
        let transformer : Transformer = null;
        for(let i=0; i<this._transformers.length; i++){
            if(this._transformers[i].getId() == id){
                transformer = this._transformers[i];
                break;
            }
        }
        return transformer;
    }
}