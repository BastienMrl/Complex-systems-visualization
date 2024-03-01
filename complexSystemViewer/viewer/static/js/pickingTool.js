export var PickingMode;
(function (PickingMode) {
    PickingMode[PickingMode["DISABLE"] = 0] = "DISABLE";
    PickingMode[PickingMode["POINT"] = 1] = "POINT";
    PickingMode[PickingMode["BOX"] = 2] = "BOX";
    // Only Convex polygon
    PickingMode[PickingMode["LASSO"] = 3] = "LASSO";
})(PickingMode || (PickingMode = {}));
