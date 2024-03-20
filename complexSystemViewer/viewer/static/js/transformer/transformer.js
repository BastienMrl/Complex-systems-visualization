import { TransformType } from "./transformType.js";
import { ShaderVariable } from "../shaderUtils.js";
export class Transformer {
    type;
    _id;
    _inputVariable;
    constructor(id, inputVariable) {
        this._id = id;
        this._inputVariable = inputVariable;
    }
    getTypeNbElements(value) {
        if (Array.isArray(value))
            return value.length;
        return 1;
    }
    getOutputName() {
        switch (this.type) {
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
                return `${ShaderVariable.ROTATION}.x`;
            case TransformType.ROTATION_Y:
                return `${ShaderVariable.ROTATION}.y`;
            case TransformType.ROTATION_Z:
                return `${ShaderVariable.ROTATION}.z`;
            case TransformType.SCALING:
                return `${ShaderVariable.SCALING}`;
        }
    }
    getTypeDeclaration(value) {
        switch (this.getTypeNbElements(value)) {
            case 1:
                return "float";
            case 2:
                return "vec2";
            case 3:
                return "vec3";
            case 4:
                return "vec4";
        }
    }
    getVariableInitialisation(value) {
        let toFloatString = function (value) {
            if (!`${value}`.includes(".")) {
                let toFloat = parseFloat(`${value}`).toFixed(2);
                return toFloat;
            }
            return value;
        };
        switch (this.getTypeNbElements(value)) {
            case 1:
                return `${toFloatString(value)}`;
            case 2:
                return `vec2(${toFloatString(value[0])}, ${toFloatString(value[1])})`;
            case 3:
                return `vec3(${toFloatString(value[0])}, ${toFloatString(value[1])}, ${toFloatString(value[2])})`;
            case 4:
                return `vec4(${toFloatString(value[0])}, ${toFloatString(value[1])}, ${toFloatString(value[2])}, ${toFloatString(value[3])})`;
        }
    }
    getParamName(paramIdx) {
        return `param_${this._id}_${paramIdx}`;
    }
    getParamDeclaration(paramIdx, value) {
        return `const ${this.getTypeDeclaration(value)} param_${this._id}_${paramIdx} = ${this.getVariableInitialisation(value)};`;
    }
    getTransformerFunctionCall(fct, paramsIdx) {
        let s = `${fct}(${this.getOutputName()}`;
        paramsIdx.forEach(e => {
            s += `, ${this.getParamName(e)}`;
        });
        s += `, ${this._inputVariable});`;
        return s;
    }
    setInputVariable(variable) {
        this._inputVariable = variable;
    }
    getInputVariable() {
        return this._inputVariable;
    }
    getId() {
        return this._id;
    }
}
