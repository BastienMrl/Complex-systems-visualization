import { ShaderFunction } from "../shaderUtils.js";
import { TransformType } from "./transformType.js";
import { Transformer } from "./transformer.js";
export class PositionTransformer extends Transformer {
    _factor;
    constructor(idx, inputVariable, axis, factor = 1.) {
        super(idx, inputVariable);
        this.setFactor(factor);
        switch (axis) {
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
    applyTransformation(input) {
        return input * this.factor;
    }
    setFactor(factor) {
        this._factor = factor;
    }
    getParamsDeclarationBlock() {
        return this.getParamDeclaration(0, this._factor);
    }
    getTransformationsBlock() {
        return this.getTransformerFunctionCall(ShaderFunction.FACTOR, [0]);
    }
    setParameters(params) {
        this.setFactor(params[0]);
    }
    get factor() {
        return this._factor;
    }
}
