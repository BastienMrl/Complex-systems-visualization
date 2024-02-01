import {mat4} from "./glMatrix/esm/index.js"

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
    

    constructor(cameraPosition, cameraTarget, up, fovy, aspect, near, far){
        this.#cameraMatrix = mat4.create();
        this.#viewMatrix = mat4.create();
        this.#projectionMatrix = mat4.create();
        this.#projViewMatrix = mat4.create();

        this.#cameraPosition = cameraPosition;
        this.#cameraTarget = cameraTarget;
        this.#up = up;
        this.#updateCameraMatrix();


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
}

export {Camera}