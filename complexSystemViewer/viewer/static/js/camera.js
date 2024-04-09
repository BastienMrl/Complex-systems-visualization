import { Mat4, Vec3 } from "./ext/glMatrix/index.js";
export class Camera {
    // projection parameters
    _fovy;
    _aspect;
    _near;
    _far;
    _projectionMatrix;
    _isOrthographic = false;
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
    constructor(cameraPosition, cameraTarget, up) {
        this._cameraMatrix = new Mat4();
        this._viewMatrix = new Mat4();
        this._projectionMatrix = new Mat4();
        this._projViewMatrix = new Mat4();
        this._cameraPosition = cameraPosition;
        this._cameraTarget = cameraTarget;
        this._up = up;
        this._distance = Vec3.distance(this._cameraPosition, this._cameraTarget);
        this.updateCameraMatrix();
    }
    static getPerspectiveCamera(cameraPosition, cameraTarget, up, fovy, aspect, near, far) {
        let camera = new Camera(cameraPosition, cameraTarget, up);
        camera._fovy = fovy;
        camera._aspect = aspect;
        camera._near = near;
        camera._far = far;
        camera.updateProjectionMatrix();
        return camera;
    }
    static getOrthographicCamera(cameraPosition, cameraTarget, up, aspect, near, far) {
        let camera = new Camera(cameraPosition, cameraTarget, up);
        camera._isOrthographic = true;
        camera._aspect = aspect;
        camera._near = near;
        camera._far = far;
        camera.updateProjectionMatrix();
        return camera;
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
    get target() {
        let copy = Vec3.create();
        Vec3.copy(copy, this._cameraTarget);
        return copy;
    }
    get distance() {
        return this._distance;
    }
    get aspectRatio() {
        return this._aspect;
    }
    get isOrthographic() {
        return this._isOrthographic;
    }
    getOrthographicBoundaries() {
        return [-this._distance * this._aspect, this._distance * this._aspect, -this._distance, this._distance];
    }
    // setters
    set aspectRatio(aspect) {
        this._aspect = aspect;
        this.updateProjectionMatrix();
    }
    set distanceMin(dst) {
        this._distanceMin = dst;
    }
    set distanceMax(dst) {
        this._distanceMax = dst;
    }
    // private methods
    updateProjectionMatrix() {
        if (this._isOrthographic) {
            this._projectionMatrix.orthoNO(-this._distance * this._aspect, this._distance * this._aspect, -this._distance, this._distance, this._near, this._far);
        }
        else {
            this._projectionMatrix.perspectiveNO(this._fovy, this._aspect, this._near, this._far);
        }
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
        if (this._isOrthographic) {
            this.updateProjectionMatrix();
        }
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
    move(deltaX, deltaZ) {
        this._cameraTarget.add(Vec3.fromValues(deltaX, 0, deltaZ));
        this._cameraPosition.add(Vec3.fromValues(deltaX, 0, deltaZ));
        this.updateCameraMatrix();
    }
}
