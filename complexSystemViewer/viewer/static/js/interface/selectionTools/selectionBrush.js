import { SelectionTool } from "./selectionTool.js";
export var BrushShape;
(function (BrushShape) {
    BrushShape[BrushShape["SQUARE"] = 0] = "SQUARE";
    BrushShape[BrushShape["CIRCLE"] = 1] = "CIRCLE";
})(BrushShape || (BrushShape = {}));
export class SelectionBrushTool extends SelectionTool {
    static _radiusMin = 0;
    static _radiusMax = 30;
    _interactionButton;
    _shape;
    _radius;
    _intensity;
    _attenuationFunction;
    _prevId;
    _idValues;
    constructor(viewer, interactionButton, manager) {
        super(viewer, manager);
        this._interactionButton = interactionButton;
        this._shape = BrushShape.CIRCLE;
        this._radius = 3;
        this._intensity = 0.4;
        this._attenuationFunction = function (distance) {
            if (distance > this._radius)
                return 0;
            let normalize = distance / (this._radius + 1);
            let z = Math.exp(-4 * Math.pow(this._intensity - 1, 2)) * (1 - Math.pow(normalize, 3));
            return z;
        };
        this._prevId = null;
        this._idValues = new Map();
        // Temporary
        window.addEventListener('keydown', e => {
            if (e.code == 'ArrowUp')
                this.setBrushRadius(this._radius + 1);
            else if (e.code == 'ArrowDown')
                this.setBrushRadius(this._radius - 1);
            else if (e.code == 'ArrowLeft' || e.code == 'ArrowRight') {
                let shape = this._shape == BrushShape.CIRCLE ? BrushShape.SQUARE : BrushShape.CIRCLE;
                this.setBrushShape(shape);
            }
        });
    }
    setBrushRadius(radius) {
        if (radius < SelectionBrushTool._radiusMin)
            this._radius = SelectionBrushTool._radiusMin;
        else if (radius > SelectionBrushTool._radiusMax)
            this._radius = SelectionBrushTool._radiusMax;
        else
            this._radius = radius;
        this.updateCurrentMouseOver();
    }
    setBrushShape(shape) {
        this._shape = shape;
        this.updateCurrentMouseOver();
    }
    onMouseMove(e) {
        if (!this._mouseDown) {
            this.updateCurrentMouseOver();
            return;
        }
        else if (this._currentId != null) {
            let path = [this._currentId];
            if (this._prevId != null)
                path = this.getPath(this._prevId, this._currentId);
            this._prevId = this._currentId;
            this.fillCurrentValues(path);
        }
        else {
            this._prevId = null;
        }
        this.onCurrentSelectionChanged(this._idValues);
    }
    updateCurrentMouseOver() {
        if (this._mouseDown)
            return;
        this._idValues.clear();
        if (this._currentId == null)
            this.onCurrentSelectionChanged(null);
        else {
            this.fillCurrentValues([this._currentId]);
            this.onCurrentSelectionChanged(this._idValues);
        }
    }
    onMouseDown(e) {
        if (e.button != this._interactionButton)
            return;
        this._mouseDown = true;
        this._prevId = this._currentId;
        this._idValues.clear();
    }
    onMouseUp(e) {
        if (e.button != this._interactionButton || !this._mouseDown)
            return;
        this._mouseDown = false;
        if (this._currentMask.length != 0)
            this._manager.apply_interaction(new Float32Array(this._currentMask));
        this.onCurrentSelectionChanged(null);
    }
    getPath(fromId, targetId) {
        if (fromId == null || !Number.isInteger(fromId))
            return [];
        if (targetId == null || !Number.isInteger(targetId))
            return [];
        const targetCoord = this.idToCoords(targetId);
        const fromCoord = this.idToCoords(fromId);
        let ret = [];
        let i0 = fromCoord[0];
        const i1 = targetCoord[0];
        let j0 = fromCoord[1];
        const j1 = targetCoord[1];
        const di = Math.abs(i1 - i0);
        const si = i0 < i1 ? 1 : -1;
        const dj = -Math.abs(j1 - j0);
        const sj = j0 < j1 ? 1 : -1;
        let error = di + dj;
        while (true) {
            ret.push(this.coordToId(i0, j0));
            if (i0 == i1 && j0 == j1)
                break;
            const e2 = 2 * error;
            if (e2 > dj) {
                error += dj;
                i0 += si;
            }
            if (e2 < di) {
                error += di;
                j0 += sj;
            }
        }
        return ret;
    }
    fillCurrentValues(path) {
        path.forEach((e) => {
            let newMap = this.getIdValuesInsideBrush(e);
            newMap.forEach((value, key, map) => {
                let currentValue = this._idValues.get(key);
                if (currentValue == undefined || currentValue < value)
                    this._idValues.set(key, value);
            });
        });
    }
    getIdValuesInsideBrush(from) {
        let ret = new Map();
        let centerCoord = this.idToCoords(from);
        let iMin = centerCoord[0] - this._radius;
        let iMax = centerCoord[0] + this._radius;
        let jMin = centerCoord[1] - this._radius;
        let jMax = centerCoord[1] + this._radius;
        if (iMin < 0)
            iMin = 0;
        if (iMax >= this._maskSize[1])
            iMax = this._maskSize[1] - 1;
        if (jMin < 0)
            jMin = 0;
        if (jMax >= this._maskSize[0])
            jMax = this._maskSize[0] - 1;
        for (let i = iMin; i < iMax + 1; i++) {
            for (let j = jMin; j < jMax + 1; j++) {
                let absI = Math.abs(centerCoord[0] - i);
                let absJ = Math.abs(centerCoord[1] - j);
                // attenuationFunction exclude this case
                let distance = this._radius + 1;
                switch (this._shape) {
                    case BrushShape.SQUARE:
                        distance = Math.max(absI, absJ);
                        break;
                    case BrushShape.CIRCLE:
                        distance = Math.sqrt(absI * absI + absJ * absJ);
                        break;
                }
                if (distance <= this._radius)
                    ret.set(this.coordToId(i, j), this._attenuationFunction(distance));
            }
        }
        return ret;
    }
    setParam(attribute, value) {
        switch (attribute) {
            case "radius":
                this._radius = value;
                break;
            case "intensity":
                this._intensity = value;
                break;
            default:
                throw Error("BAD ATTRIBUTE SELECTION IN BRUSH SELECTOR WITH : " + attribute);
        }
    }
    getAllParam() {
        return JSON.stringify({
            "intensity": {
                "min": 0,
                "max": 1,
                "step": 0.01,
                "value": 0.5,
                "type": "range"
            },
            "radius": {
                "min": SelectionBrushTool._radiusMin,
                "max": SelectionBrushTool._radiusMax,
                "step": 1,
                "value": 3,
                "type": "range"
            }
        });
    }
}
