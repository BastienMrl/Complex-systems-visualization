export var TransformType;
(function (TransformType) {
    TransformType[TransformType["COLOR"] = 0] = "COLOR";
    TransformType[TransformType["COLOR_R"] = 1] = "COLOR_R";
    TransformType[TransformType["COLOR_G"] = 2] = "COLOR_G";
    TransformType[TransformType["COLOR_B"] = 3] = "COLOR_B";
    TransformType[TransformType["POSITION_X"] = 4] = "POSITION_X";
    TransformType[TransformType["POSITION_Y"] = 5] = "POSITION_Y";
    TransformType[TransformType["POSITION_Z"] = 6] = "POSITION_Z";
    TransformType[TransformType["ROTATION_X"] = 7] = "ROTATION_X";
    TransformType[TransformType["ROTATION_Y"] = 8] = "ROTATION_Y";
    TransformType[TransformType["ROTATION_Z"] = 9] = "ROTATION_Z";
    TransformType[TransformType["SCALING"] = 10] = "SCALING";
})(TransformType || (TransformType = {}));
export var TransformFlag;
(function (TransformFlag) {
    TransformFlag[TransformFlag["ALL"] = 1] = "ALL";
    TransformFlag[TransformFlag["COLOR"] = 2] = "COLOR";
    TransformFlag[TransformFlag["POSITION"] = 4] = "POSITION";
    TransformFlag[TransformFlag["ROTATION"] = 8] = "ROTATION";
    TransformFlag[TransformFlag["SCALING"] = 16] = "SCALING";
})(TransformFlag || (TransformFlag = {}));
export function transformTypeMatchFlag(type, flag) {
    switch (type) {
        case TransformType.COLOR:
        case TransformType.COLOR_R:
        case TransformType.COLOR_G:
        case TransformType.COLOR_B:
            return 0 < (flag & (TransformFlag.ALL | TransformFlag.COLOR));
        case TransformType.POSITION_X:
        case TransformType.POSITION_Y:
        case TransformType.POSITION_Z:
            return 0 < (flag & (TransformFlag.ALL | TransformFlag.POSITION));
        case TransformType.ROTATION_X:
        case TransformType.ROTATION_Y:
        case TransformType.ROTATION_Z:
            return 0 < (flag & (TransformFlag.ALL | TransformFlag.ROTATION));
        case TransformType.SCALING:
            return 0 < (flag & (TransformFlag.ALL | TransformFlag.SCALING));
    }
}
