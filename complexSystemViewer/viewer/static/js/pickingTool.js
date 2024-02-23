import { Vec3 } from "./ext/glMatrix/vec3.js";
export class PickingTool {
    _meshes;
    _transformer;
    _viewer;
    constructor(viewer) {
        this._viewer = viewer;
    }
    setMeshes(meshes) {
        this._meshes = meshes;
    }
    setTransformer(transformer) {
        this._transformer = transformer;
    }
    getMeshesId(mouseX, mouseY, width, height, camera) {
        let origin = camera.position;
        let x = (2.0 * mouseX) / width - 1.0;
        let y = 1.0 - (2.0 * mouseY) / height;
        let z = 1.0;
        let direction = Vec3.fromValues(x, y, z);
        Vec3.transformMat4(direction, direction, camera.projViewMatrix.invert());
        direction.normalize();
        let normal = Vec3.fromValues(0, 1, 0);
        let pFar = Vec3.fromValues(0, -0.5, 0);
        let pNear = Vec3.fromValues(0, 0.5, 0);
        let denominator = Vec3.dot(normal, direction);
        let tFar = -1;
        let tNear = -1;
        if (denominator != 0) {
            let p = Vec3.create();
            Vec3.sub(p, pFar, origin);
            tFar = Vec3.dot(p, normal) / denominator;
            Vec3.sub(p, pNear, origin);
            tNear = Vec3.dot(p, normal) / denominator;
        }
        if (tFar < 0 || tNear < 0)
            return null;
        const nbSample = 10;
        const tDelta = tFar - tNear;
        for (let i = 0; i < nbSample; i++) {
            let step = (i) / (nbSample - 1);
            let t = tNear + tDelta * step;
            let position = Vec3.create();
            direction.scale(t);
            Vec3.add(position, origin, direction);
            let id = this.getMeshId(position[0], position[2]);
            if (id != null)
                return id;
        }
        return null;
    }
    getMeshId(x, z) {
        let offsetX = (this._meshes.nbCol - 1);
        let offsetZ = (this._meshes.nbRow - 1);
        let xMin = (x - 0.5) / this._transformer.getPositionFactor(0);
        let xMax = (x + 0.5) / this._transformer.getPositionFactor(0);
        let zMin = (z - 0.5) / this._transformer.getPositionFactor(2);
        let zMax = (z + 0.5) / this._transformer.getPositionFactor(2);
        xMin = (Math.round(xMin * 2) + offsetX) / 2;
        xMax = (Math.round(xMax * 2) + offsetX) / 2;
        zMin = (Math.round(zMin * 2) + offsetZ) / 2;
        zMax = (Math.round(zMax * 2) + offsetZ) / 2;
        if (xMin == xMax)
            x = Math.round(xMax);
        else if (Number.isInteger(xMin) && Number.isInteger(xMax))
            x = xMax;
        else if (Math.ceil(xMin) == Math.floor(xMax))
            x = Math.ceil(xMin);
        if (zMin == zMax)
            z = Math.round(zMax);
        else if (Number.isInteger(zMin) && Number.isInteger(zMax))
            z = zMax;
        else if (Math.ceil(zMin) == Math.floor(zMax))
            z = Math.ceil(zMin);
        if (z >= this._meshes.nbRow || z < 0 || x >= this._meshes.nbCol || x < 0)
            return null;
        let id = z * this._meshes.nbCol + x;
        return id;
    }
}
