import { ShaderFunction } from "../shaderUtils.js";
import * as Utils from "../typeUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";
export class ColorTransformer extends Transformer {
    type = TransformType.COLOR;
    _colorMin;
    _colorMax;
    constructor(id, inputVariable, params) {
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
    applyTransformation(input) {
        let ret = this._colorMin.map(x => x * (1 - input));
        return ret.map((x, i) => x + this._colorMax[i] * input);
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
            this._colorMin = Utils.hexToRgbA(params[0]);
        if (typeof params[1] == "string")
            this._colorMax = Utils.hexToRgbA(params[1]);
    }
}
