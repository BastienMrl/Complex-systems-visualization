import {mat4, vec3} from "./glMatrix/esm/index.js"
import { fromValues } from "./glMatrix/esm/mat2.js";
import { max } from "./glMatrix/esm/vec3.js";

class Camera{

    // projection parameters
    #fovy;
    #aspect;
    #near;
    #far;
    #projectionMatrix;

    // camera parameters
    #cameraPosition;
    #cameraTarget;
    #up;
    #cameraMatrix;
    #viewMatrix;

    #projViewMatrix;

    //trackball
    #distanceMin = 4.
    #distanceMax = 5000.
    #distance
    #angleX = - Math.PI / 4;
    #angleY = 0
    #minAngleX = - Math.PI / 2;
    #maxAngleX = - Math.PI / 16;
    

    constructor(cameraPosition, cameraTarget, up, fovy, aspect, near, far){
        this.#cameraMatrix = mat4.create();
        this.#viewMatrix = mat4.create();
        this.#projectionMatrix = mat4.create();
        this.#projViewMatrix = mat4.create();

        this.#cameraPosition = cameraPosition;
        this.#cameraTarget = cameraTarget;
        this.#up = up;
        this.#updateCameraMatrix();
        this.#distance = vec3.distance(this.#cameraPosition, this.#cameraTarget);


        this.#fovy = fovy;
        this.#aspect = aspect;
        this.#near = near;
        this.#far = far;
        
        this.#updateProjectionMatrix();
    }

    #updateProjectionMatrix(){
        mat4.perspective(this.#projectionMatrix, this.#fovy, this.#aspect, this.#near, this.#far);
        this.#updateProjViewMatrix()
    }

    getProjectionMatrix(){
        return this.#projectionMatrix;
    }

    #updateCameraMatrix(){
        mat4.targetTo(this.#cameraMatrix, this.#cameraPosition, this.#cameraTarget, this.#up);
        mat4.invert(this.#viewMatrix, this.#cameraMatrix);
        this.#updateProjViewMatrix()
    }
    
    getViewMatrix(){
        return this.#viewMatrix;
    }

    #updateProjViewMatrix(){
        mat4.multiply(this.#projViewMatrix, this.#projectionMatrix, this.#viewMatrix);
    }

    getProjViewMatrix() {
        return this.#projViewMatrix;
    }

    moveForward(distance){
        distance *= this.#distance;
        distance = this.#distanceMin < (this.#distance - distance) ? distance : this.#distance - this.#distanceMin;
        distance = this.#distanceMax > (this.#distance - distance) ? distance : 0;

        let dir = vec3.create()
        vec3.sub(dir, this.#cameraTarget, this.#cameraPosition);
        vec3.normalize(dir, dir);
        vec3.scale(dir, dir, distance);
        vec3.add(this.#cameraPosition, this.#cameraPosition, dir);
        this.#distance -= distance;
        this.#updateCameraMatrix();
    }

    rotateCamera(deltaX, deltaY){

        let matrix = mat4.create();      

        this.#angleY -= deltaY % (2 * Math.PI);
        this.#angleX -= deltaX
        this.#angleX = Math.min(this.#maxAngleX, this.#angleX);
        this.#angleX = Math.max(this.#minAngleX, this.#angleX);
        
        mat4.fromYRotation(matrix, this.#angleY);
        mat4.rotateX(matrix, matrix, this.#angleX);
        mat4.translate(matrix, matrix, vec3.fromValues(0, 0, this.#distance));       

        vec3.transformMat4(this.#cameraPosition, vec3.fromValues(0, 0, 0), matrix);

        this.#updateCameraMatrix();
    }
}

export {Camera}