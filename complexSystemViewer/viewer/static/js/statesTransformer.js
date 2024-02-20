import { ShaderVariable, ShaderFunction, ShaderMeshInputs, ShaderUniforms } from "./shaderUtils.js";
export var TransformType;
(function (TransformType) {
    TransformType[TransformType["COLOR"] = 0] = "COLOR";
    TransformType[TransformType["COLOR_R"] = 1] = "COLOR_R";
    TransformType[TransformType["COLOR_G"] = 2] = "COLOR_G";
    TransformType[TransformType["COLOR_B"] = 3] = "COLOR_B";
    TransformType[TransformType["POSITION_X"] = 4] = "POSITION_X";
    TransformType[TransformType["POSITION_Y"] = 5] = "POSITION_Y";
    TransformType[TransformType["POSITION_Z"] = 6] = "POSITION_Z";
})(TransformType || (TransformType = {}));
export var InputType;
(function (InputType) {
    InputType[InputType["POSITION_X"] = 0] = "POSITION_X";
    InputType[InputType["POSITION_Y"] = 1] = "POSITION_Y";
    InputType[InputType["POSITION_Z"] = 2] = "POSITION_Z";
    InputType[InputType["STATE_0"] = 3] = "STATE_0";
    InputType[InputType["STATE_1"] = 4] = "STATE_1";
    InputType[InputType["STATE_2"] = 5] = "STATE_2";
    InputType[InputType["STATE_3"] = 6] = "STATE_3";
    InputType[InputType["STATE_4"] = 7] = "STATE_4";
    InputType[InputType["STATE_5"] = 8] = "STATE_5";
    InputType[InputType["STATE_6"] = 9] = "STATE_6";
    InputType[InputType["STATE_7"] = 10] = "STATE_7";
    InputType[InputType["STATE_8"] = 11] = "STATE_8";
    InputType[InputType["STATE_9"] = 12] = "STATE_9";
})(InputType || (InputType = {}));
export class StatesTransformer {
    _transformers;
    _dataIndices;
    _idCpt;
    _inputDeclarations;
    constructor() {
        this._transformers = [];
        this._dataIndices = [];
        this._idCpt = 0;
        this._inputDeclarations = [];
    }
    addInputVariableDeclaration(transformType, intputType, name) {
        let s = `float ${name} = `;
        let onT0 = "";
        let onT1 = "";
        let time = "";
        let normalized = false;
        let need_normalization = false;
        let normalization_axis = null;
        switch (transformType) {
            case TransformType.COLOR:
                time = ShaderUniforms.TIME_COLOR;
                need_normalization = true;
                break;
            case TransformType.COLOR_R:
                time = ShaderUniforms.TIME_COLOR;
                need_normalization = true;
                break;
            case TransformType.COLOR_G:
                time = ShaderUniforms.TIME_COLOR;
                need_normalization = true;
                break;
            case TransformType.COLOR_B:
                time = ShaderUniforms.TIME_COLOR;
                need_normalization = true;
                break;
            case TransformType.POSITION_X:
                time = ShaderUniforms.TIME_TRANSLATION;
                break;
            case TransformType.POSITION_Y:
                time = ShaderUniforms.TIME_TRANSLATION;
                break;
            case TransformType.POSITION_Z:
                time = ShaderUniforms.TIME_TRANSLATION;
                break;
        }
        switch (intputType) {
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
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_1:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_2:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_3:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_4:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_5:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_6:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_7:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_8:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
            case InputType.STATE_9:
                onT0 = ShaderMeshInputs.STATE_T0;
                onT1 = ShaderMeshInputs.STATE_T1;
                break;
        }
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
                s += "c";
                break;
            case TransformType.COLOR_R:
                s += "c";
                break;
            case TransformType.COLOR_G:
                s += "c";
                break;
            case TransformType.COLOR_B:
                s += "c";
                break;
            case TransformType.POSITION_X:
                s += "t";
                break;
            case TransformType.POSITION_Y:
                s += "t";
                break;
            case TransformType.POSITION_Z:
                s += "t";
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
                s += "s";
                break;
            case InputType.STATE_1:
                s += "s";
                break;
            case InputType.STATE_2:
                s += "s";
                break;
            case InputType.STATE_3:
                s += "s";
                break;
            case InputType.STATE_4:
                s += "s";
                break;
            case InputType.STATE_5:
                s += "s";
                break;
            case InputType.STATE_6:
                s += "s";
                break;
            case InputType.STATE_7:
                s += "s";
                break;
            case InputType.STATE_8:
                s += "s";
                break;
            case InputType.STATE_9:
                s += "s";
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
                this._dataIndices.push(inputType);
                break;
            case TransformType.COLOR_R:
                break;
            case TransformType.COLOR_G:
                break;
            case TransformType.COLOR_B:
                break;
            case TransformType.POSITION_X:
                this._transformers.push(new PositionTransformer(id, inputVariable, 0, params == undefined ? 1. : params[0]));
                this._dataIndices.push(inputType);
                break;
            case TransformType.POSITION_Y:
                this._transformers.push(new PositionTransformer(id, inputVariable, 1, params == undefined ? 1. : params[0]));
                this._dataIndices.push(inputType);
                break;
            case TransformType.POSITION_Z:
                this._transformers.push(new PositionTransformer(id, inputVariable, 2, params == undefined ? 1. : params[0]));
                this._dataIndices.push(inputType);
                break;
        }
        this.addInputVariableDeclaration(type, inputType, inputVariable);
        return this._transformers.length - 1;
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
        if (id < 0 || id >= this._transformers.length)
            return;
        this._transformers[id].setParameters(params);
    }
    setInputType(id, inputType) {
        if (id < 0 || id >= this._transformers.length)
            return;
        let oldVariable = this._transformers[id].getInputVariable();
        let transformType = this._transformers[id].type;
        let newVariable = this.getInputVariableName(transformType, inputType);
        this.addInputVariableDeclaration(transformType, inputType, newVariable);
        this._transformers[id].setInputVariable(newVariable);
        this.deleteVariableDeclaration(oldVariable);
    }
}
class Transformer {
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
        switch (this.getTypeNbElements(value)) {
            case 1:
                return `${value}`;
            case 2:
                return `vec2(${value[0]}, ${value[1]})`;
            case 3:
                return `vec3(${value[0]}, ${value[1]}, ${value[2]})`;
            case 4:
                return `vec4(${value[0]}, ${value[1]}, ${value[2]}, ${value[3]})`;
        }
    }
    getParamName(paramIdx) {
        return `param_${this._id}_${paramIdx}`;
    }
    getParamDeclaration(paramIdx, value) {
        return `const ${this.getTypeDeclaration(value)} param_${this._id}_${paramIdx} = ${this.getVariableInitialisation(value)};`;
    }
    getTransformerFunctionCall(fct, params) {
        let s = `${fct}(${this.getOutputName()}`;
        params.forEach(e => {
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
}
class ColorTransformer extends Transformer {
    type = TransformType.COLOR;
    _colorMin;
    _colorMax;
    _nbParams = 2;
    constructor(id, inputVariable, params) {
        super(id, inputVariable);
        if (typeof params[0] == "string")
            this._colorMin = this.hexToRgbA(params[0]);
        else
            this._colorMin = [0., 0., 0.];
        if (typeof params[1] == "string")
            this._colorMax = this.hexToRgbA(params[1]);
        else
            this._colorMax = [1., 1., 1.];
    }
    hexToRgbA(hex) {
        let c;
        if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
            c = hex.substring(1).split('');
            if (c.length == 3) {
                c = [c[0], c[0], c[1], c[1], c[2], c[2]];
            }
            c = '0x' + c.join('');
            return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
        }
        throw new Error('Bad Hex');
    }
    getParamsDeclarationBlock() {
        let s = "";
        s += this.getParamDeclaration(0, this._colorMin) + "\n";
        s += this.getParamDeclaration(1, this._colorMax);
        return s;
    }
    getTransformationsBlock() {
        return this.getTransformerFunctionCall(ShaderFunction.INTERPOLATION, [0, 1]);
    }
    setParameters(params) {
        if (typeof params[0] == "string")
            this._colorMin = this.hexToRgbA(params[0]);
        if (typeof params[1] == "string")
            this._colorMax = this.hexToRgbA(params[1]);
    }
}
class PositionTransformer extends Transformer {
    _factor;
    constructor(idx, inputVariable, axe, factor = 1.) {
        super(idx, inputVariable);
        this._factor = factor;
        switch (axe) {
            case 0:
                this.type = TransformType.POSITION_X;
                break;
            case 1:
                this.type = TransformType.POSITION_Y;
                break;
            case 2:
                this.type = TransformType.POSITION_Z;
                break;
        }
    }
    getParamsDeclarationBlock() {
        return this.getParamDeclaration(0, this._factor);
    }
    getTransformationsBlock() {
        return this.getTransformerFunctionCall(ShaderFunction.FACTOR, [0]);
    }
    setParameters(...args) {
        this._factor = args[0];
    }
}
