export enum TransformType {
    COLOR,
    COLOR_R,
    COLOR_G,
    COLOR_B,

    POSITION_X,
    POSITION_Y,
    POSITION_Z,

    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,

    SCALING
}

export enum TransformFlag {
    ALL = 1,
    COLOR = 2,
    POSITION = 4,
    ROTATION = 8,
    SCALING = 16
}

export function transformTypeMatchFlag(type : TransformType, flag : TransformFlag) : boolean{
    switch (type){
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

