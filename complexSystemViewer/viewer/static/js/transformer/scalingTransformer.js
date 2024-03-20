import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";
export class ScalingTransformer extends Transformer {
    _min;
    _max;
    type = TransformType.SCALING;
    constructor(idx, inputVariable, min, max) {
        super(idx, inputVariable);
        this._min = min;
        this._max = max;
    }
    applyTransformation(input) {
        return this._min * (1 - input) + this._max * input;
    }
    getParamsDeclarationBlock() {
        let s = "";
        s += this.getParamDeclaration(0, this._min) + "\n";
        s += this.getParamDeclaration(1, this._max);
        return s;
    }
    getTransformationsBlock() {
        return this.getTransformerFunctionCall(ShaderFunction.INTERPOLATION, [0, 1]);
    }
    setParameters(params) {
        if (params[0] != null) {
            this._min = params[0];
        }
        if (params[1] != null) {
            this._max = params[1];
        }
    }
}
