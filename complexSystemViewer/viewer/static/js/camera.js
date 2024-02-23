import { Mat4, Vec3 } from "./ext/glMatrix/index.js";
export class Camera {
    // projection parameters
    _fovy;
    _aspect;
    _near;
    _far;
    _projectionMatrix;
    // camera parameters
    _cameraPosition;
    _cameraTarget;
    _up;
    _cameraMatrix;
    _viewMatrix;
    _projViewMatrix;
    //trackball
    _distanceMin = 4.;
    _distanceMax = 5000.;
    _distance;
    _angleX = -Math.PI / 4;
    _angleY = 0;
    _minAngleX = -Math.PI / 2;
    _maxAngleX = -Math.PI / 16;
    constructor(cameraPosition, cameraTarget, up, fovy, aspect, near, far) {
        this._cameraMatrix = new Mat4();
        this._viewMatrix = new Mat4();
        this._projectionMatrix = new Mat4();
        this._projViewMatrix = new Mat4();
        this._fovy = fovy;
        this._aspect = aspect;
        this._near = near;
        this._far = far;
        this.updateProjectionMatrix();
        this._cameraPosition = cameraPosition;
        this._cameraTarget = cameraTarget;
        this._up = up;
        this._distance = Vec3.distance(this._cameraPosition, this._cameraTarget);
        this.updateCameraMatrix();
    }
    // getters
    get projectionMatrix() {
        let copy = Mat4.create();
        Mat4.copy(copy, this._projectionMatrix);
        return copy;
    }
    get viewMatrix() {
        let copy = Mat4.create();
        Mat4.copy(copy, this._viewMatrix);
        return copy;
    }
    get projViewMatrix() {
        let copy = Mat4.create();
        Mat4.copy(copy, this._projViewMatrix);
        return copy;
    }
    get position() {
        let copy = Vec3.create();
        Vec3.copy(copy, this._cameraPosition);
        return copy;
    }
    // setters
    set aspectRatio(aspect) {
        this._aspect = aspect;
        this.updateProjectionMatrix();
    }
    // private methods
    updateProjectionMatrix() {
        this._projectionMatrix.perspectiveNO(this._fovy, this._aspect, this._near, this._far);
        this.updateProjViewMatrix();
    }
    updateCameraMatrix() {
        Mat4.targetTo(this._cameraMatrix, this._cameraPosition, this._cameraTarget, this._up);
        Mat4.invert(this._viewMatrix, this._cameraMatrix);
        this.updateProjViewMatrix();
    }
    updateProjViewMatrix() {
        Mat4.multiply(this._projViewMatrix, this._projectionMatrix, this._viewMatrix);
    }
    // public methods
    moveForward(distance) {
        distance *= this._distance;
        distance = this._distanceMin < (this._distance - distance) ? distance : this._distance - this._distanceMin;
        distance = this._distanceMax > (this._distance - distance) ? distance : 0;
        let dir = new Vec3();
        Vec3.sub(dir, this._cameraTarget, this._cameraPosition);
        dir.normalize();
        dir.scale(distance);
        this._cameraPosition.add(dir);
        this._distance -= distance;
        this.updateCameraMatrix();
    }
    rotateCamera(deltaX, deltaY) {
        this._angleY -= deltaY % (2 * Math.PI);
        this._angleX -= deltaX;
        this._angleX = Math.min(this._maxAngleX, this._angleX);
        this._angleX = Math.max(this._minAngleX, this._angleX);
        let transform = new Mat4();
        Mat4.fromYRotation(transform, this._angleY);
        transform.rotateX(this._angleX);
        transform.translate(Vec3.fromValues(0., 0., this._distance));
        Vec3.transformMat4(this._cameraPosition, Vec3.fromValues(0., 0., 0.), transform);
        this.updateCameraMatrix();
    }
}
