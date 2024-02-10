import { Vec3 } from "./glMatrix/index.js";
const sizePerColor = 3;
const sizePerTranslation = 3;
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
export class TransformableValues {
    colors;
    translations;
    constructor(nbElements) {
        this.colors = new Float32Array(nbElements * sizePerColor).fill(0);
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0);
    }
}
export class StatesTransformer {
    _transformers;
    _dataIndices;
    constructor() {
        this._transformers = [];
        this._dataIndices = [];
    }
    addTransformer(type, dataIndex, ...args) {
        switch (type) {
            case TransformType.COLOR:
                this._transformers.push(new ColorTransformer(args[0], args[1]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.COLOR_R:
                break;
            case TransformType.COLOR_G:
                break;
            case TransformType.COLOR_B:
                break;
            case TransformType.POSITION_X:
                this._transformers.push(new PositionTransformer(0, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.POSITION_Y:
                this._transformers.push(new PositionTransformer(1, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
            case TransformType.POSITION_Z:
                this._transformers.push(new PositionTransformer(2, args[0] == undefined ? 1. : args[0]));
                this._dataIndices.push(dataIndex);
                break;
        }
        return this._transformers.length - 1;
    }
    applyTransformers(data, values) {
        this._transformers.forEach((transformer, idx) => {
            transformer.transform(data[this._dataIndices[idx]], values);
        });
    }
    setParams(id, ...args) {
        if (id < 0 || id >= this._transformers.length)
            return;
        this._transformers[id].setParameters(args);
    }
}
class Transformer {
    type;
}
class ColorTransformer extends Transformer {
    type = TransformType.COLOR;
    _colorMin;
    _colorMax;
    constructor(colorMin, colorMax) {
        super();
        this._colorMin = Vec3.fromValues(colorMin[0], colorMin[1], colorMin[2]);
        this._colorMax = Vec3.fromValues(colorMax[0], colorMax[1], colorMax[2]);
    }
    transform(states, values) {
        states.forEach((alpha, idx) => {
            let color = new Vec3().copy(this._colorMin).scale(1 - alpha);
            color.add(new Vec3().copy(this._colorMax).scale(alpha));
            for (let i = 0; i < 3; ++i)
                values.colors[idx * 3 + i] = color[i];
        });
    }
    setParameters(...args) {
        this._colorMin = Vec3.fromValues(args[0][0], args[0][1], args[0][2]);
        this._colorMax = Vec3.fromValues(args[1][0], args[1][1], args[1][2]);
    }
}
class PositionTransformer extends Transformer {
    _axeIdx;
    _factor;
    constructor(idx, factor = 1.) {
        super();
        this._axeIdx = idx;
        this._factor = factor;
        switch (idx) {
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
    transform(states, values) {
        states.forEach((value, idx) => {
            let outIndex = idx * 3 + this._axeIdx;
            values.translations[outIndex] = value * this._factor;
        });
    }
    setParameters(...args) {
        this._factor = args[0];
    }
}
