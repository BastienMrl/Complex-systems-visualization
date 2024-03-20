import { ShaderFunction } from "../shaderUtils.js";
import * as Utils from "../typeUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";
export class ColorChannelTransformer extends Transformer {
    _min;
    _max;
    constructor(idx, inputVariable, channel, min = 0, max = 1) {
        super(idx, inputVariable);
        this._min = Utils.mapValue(0, 255, 0, 1, min);
        this._max = Utils.mapValue(0, 255, 0, 1, max);
        switch (channel) {
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
        if (params[0] != null)
            this._min = Utils.mapValue(0, 255, 0, 1, params[0]);
        if (params[1] != null)
            this._max = Utils.mapValue(0, 255, 0, 1, params[1]);
    }
}
