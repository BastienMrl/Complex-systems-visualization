import { Viewer } from "./viewer.js"

class UserEventHandler {
    // Singleton
    static #instance;

    #viewer;

    constructor(v) {
        if (UserEventHandler.#instance) {
            return UserEventHandler.#instance
        }
        UserEventHandler.#instance = this;
        this.#viewer = v;
        return this
    }

    initHandlers(){
        // userActions 
        let ctrlPressed = false;

        // LeftMouseButtonDown
        this.#viewer.canvas.addEventListener('mousedown', (e) =>{
            if (e.button == 0)
                console.log("leftMousePressed")
        });

        // LeftMouseButtonUp
        this.#viewer.canvas.addEventListener('mouseup', (e) => {
            if (e.button == 0)
                console.log("leftMouseUp")
        });

        // KeyDown
        window.addEventListener('keydown', (e) => {
            if (e.key == "Control"){
                ctrlPressed = true;
            }
        })

        // KeyUp
        window.addEventListener('keyup', (e) => {
            if (e.key == "Control"){
                ctrlPressed = false;
                
            }
        })

        //zoomIn/zoomOut
        this.#viewer.canvas.addEventListener('wheel', (e) =>{
            let delta = e.deltaY * 0.001;
            this.#viewer.camera.moveForward(-delta);
        });
        
        

        this.#viewer.canvas.addEventListener('mousemove', (e) => {
            if (ctrlPressed)
                this.#viewer.camera.rotateCamera(e.movementY * 0.005, e.movementX * 0.005);
        })
        //....................................................
    }
}


export { UserEventHandler }