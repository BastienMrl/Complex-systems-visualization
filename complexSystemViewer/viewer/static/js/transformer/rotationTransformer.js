import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";
export class RotationTransformer extends Transformer {
    _min;
    _max;
    constructor(idx, inputVariable, channel, min, max) {
        super(idx, inputVariable);
        switch (channel) {
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
            this._min = this.degToRad(params[0]);
        }
        if (params[1] != null) {
            this._max = this.degToRad(params[1]);
        }
    }
    degToRad(value) {
        return value * Math.PI / 180;
    }
}
