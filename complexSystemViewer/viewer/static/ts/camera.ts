import { Mat4, Vec3 } from "./ext/glMatrix/index.js"

export class Camera{

    // projection parameters
    private _fovy : number;
    private _aspect : number;
    private _near : number;
    private _far : number;
    private _projectionMatrix : Mat4;

    // camera parameters
    private _cameraPosition : Vec3;
    private _cameraTarget : Vec3;
    private _up : Vec3;
    private _cameraMatrix : Mat4;
    private _viewMatrix : Mat4;
    private _projViewMatrix : Mat4;

    //trackball
    private _distanceMin : number = 4.;
    private _distanceMax : number = 5000.;
    private _distance : number;
    private _angleX : number = - Math.PI / 4;
    private _angleY : number = 0;
    private _minAngleX :number = - Math.PI / 2;
    private _maxAngleX :number = - Math.PI / 16;
    

    public constructor(cameraPosition : Vec3, cameraTarget : Vec3, up : Vec3, fovy : number, aspect : number, near : number, far : number){
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
    public get projectionMatrix() : Mat4 {
        let copy = Mat4.create();
        Mat4.copy(copy, this._projectionMatrix);
        return copy;
    }

    public get viewMatrix() : Mat4 {
        let copy = Mat4.create();
        Mat4.copy(copy, this._viewMatrix);
        return copy;
    }
    
    public get projViewMatrix() : Mat4 {
        let copy = Mat4.create();
        Mat4.copy(copy, this._projViewMatrix);
        return copy;
    }

    public get position() : Vec3 {
        let copy = Vec3.create();
        Vec3.copy(copy, this._cameraPosition);
        return copy;
    }

    // setters
    public set aspectRatio(aspect : number){
        this._aspect = aspect;
        this.updateProjectionMatrix();
    }

    // private methods
    private updateProjectionMatrix(){
        this._projectionMatrix.perspectiveNO(this._fovy, this._aspect, this._near, this._far);
        this.updateProjViewMatrix()
    }

    private updateCameraMatrix(){
        Mat4.targetTo(this._cameraMatrix, this._cameraPosition, this._cameraTarget, this._up);
        Mat4.invert(this._viewMatrix, this._cameraMatrix);
        this.updateProjViewMatrix()
    }
    
    private updateProjViewMatrix(){
        Mat4.multiply(this._projViewMatrix, this._projectionMatrix, this._viewMatrix);
    }


    // public methods
    public moveForward(distance : number){
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

    public rotateCamera(deltaX : number, deltaY : number){
        this._angleY -= deltaY % (2 * Math.PI);
        this._angleX -= deltaX
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