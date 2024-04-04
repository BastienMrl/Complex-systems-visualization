import { ShaderFunction, ShaderMeshInputs, ShaderUniforms } from "../shaderUtils.js";
import { InputType } from "./inputType.js";
import { TransformType } from "./transformType.js";
import { ColorTransformer } from "./colorTransformer.js";
import { ColorChannelTransformer } from "./colorChannelTransformer.js";
import { ScalingTransformer } from "./scalingTransformer.js";
import { PositionTransformer } from "./positionTransformer.js";
import { RotationTransformer } from "./rotationTransformer.js";
export class TransformerBuilder {
    _transformers;
    _idCpt;
    _inputDeclarations;
    constructor() {
        this._transformers = [];
        this._idCpt = 0;
        this._inputDeclarations = [];
    }
    addInputVariableDeclaration(transformType, intputType, name) {
        let s = `float ${name} = `;
        let onT0 = "texelFetch(";
        let onT1 = "texelFetch(";
        let time = "";
        let normalized = false;
        let need_normalization = false;
        let normalization_axis = null;
        switch (transformType) {
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
                time = ShaderUniforms.TIME_SCALING;
                break;
            case TransformType.ROTATION_X:
            case TransformType.ROTATION_Y:
            case TransformType.ROTATION_Z:
                time = ShaderUniforms.TIME_ROTATION;
                break;
        }
        switch (intputType) {
            case InputType.POSITION_X:
                onT0 += ShaderMeshInputs.TEX_POS_X_T0;
                onT1 += ShaderMeshInputs.TEX_POS_X_T1;
                normalized = true && need_normalization;
                normalization_axis = 0;
                break;
            case InputType.POSITION_Y:
                onT0 += ShaderMeshInputs.TEX_POS_Y_T0;
                onT1 += ShaderMeshInputs.TEX_POS_Y_T1;
                normalized = true && need_normalization;
                normalization_axis = 2;
                break;
            case InputType.POSITION_Z:
                onT0 += ShaderMeshInputs.TEX_POS_X_T0;
                onT1 += ShaderMeshInputs.TEX_POS_X_T1;
                normalized = true && need_normalization;
                normalization_axis = 1;
                break;
            case InputType.STATE_0:
                onT0 += ShaderMeshInputs.TEX_STATE_0_T0;
                onT1 += ShaderMeshInputs.TEX_STATE_0_T1;
                break;
            case InputType.STATE_1:
                onT0 += ShaderMeshInputs.TEX_STATE_0_T0;
                onT1 += ShaderMeshInputs.TEX_STATE_0_T1;
                break;
            case InputType.STATE_2:
                onT0 += ShaderMeshInputs.TEX_STATE_0_T0;
                onT1 += ShaderMeshInputs.TEX_STATE_0_T1;
                break;
            case InputType.STATE_3:
                onT0 += ShaderMeshInputs.TEX_STATE_0_T0;
                onT1 += ShaderMeshInputs.TEX_STATE_0_T1;
                break;
        }
        onT0 += `, ${ShaderMeshInputs.UV}, 0).r`;
        onT1 += `, ${ShaderMeshInputs.UV}, 0).r`;
        s += `mix(${onT0}, ${onT1}, ${time});`;
        if (normalized)
            s += `\n${ShaderFunction.NORMALIZE_POSITION}(${name}, ${normalization_axis});`;
        this._inputDeclarations.push(s);
    }
    deleteVariableDeclaration(variable) {
        let idx = -1;
        for (let i = 0; i < this._inputDeclarations.length; ++i) {
            if (this._inputDeclarations[i].includes(`${variable}`))
                idx = i;
        }
        if (idx >= 0)
            this._inputDeclarations.splice(idx, 1);
    }
    getInputVariableName(transformType, intputType) {
        let s = "input_";
        switch (transformType) {
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
        switch (intputType) {
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
    addTransformer(type, inputType, params) {
        let inputVariable = this.getInputVariableName(type, inputType);
        let id = this._idCpt++;
        switch (type) {
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
    removeTransformer(id) {
        let transformer = this.getTransformerFromId(id);
        if (transformer == null)
            return;
        let variable = transformer.getInputVariable();
        this.deleteVariableDeclaration(variable);
        this._transformers.splice(this._transformers.indexOf(transformer), 1);
    }
    generateTransformersBlock() {
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
    generateTranslationTransformersBlock() {
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
                t == TransformType.POSITION_Z) {
                constants += transformer.getParamsDeclarationBlock() + "\n";
                fctCalls += transformer.getTransformationsBlock() + "\n";
            }
        });
        return `${inputDeclarations}\n${constants}\n${fctCalls}`;
    }
    setParams(id, params) {
        let transformer = this.getTransformerFromId(id);
        if (transformer == null)
            return;
        transformer.setParameters(params);
    }
    setInputType(id, inputType) {
        let transformer = this.getTransformerFromId(id);
        if (transformer == null)
            return;
        let oldVariable = transformer.getInputVariable();
        let transformType = transformer.type;
        let newVariable = this.getInputVariableName(transformType, inputType);
        this.addInputVariableDeclaration(transformType, inputType, newVariable);
        transformer.setInputVariable(newVariable);
        this.deleteVariableDeclaration(oldVariable);
    }
    getPositionFactor(axis) {
        let type = TransformType.POSITION_X;
        switch (axis) {
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
                factor += e.factor;
        });
        return factor;
    }
    getTransformerFromId(id) {
        let transformer = null;
        for (let i = 0; i < this._transformers.length; i++) {
            if (this._transformers[i].getId() == id) {
                transformer = this._transformers[i];
                break;
            }
        }
        return transformer;
    }
}
