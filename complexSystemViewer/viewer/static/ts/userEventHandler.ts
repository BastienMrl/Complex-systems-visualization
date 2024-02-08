import { Viewer } from "./viewer.js";

class UserEventHandler {
    // Singleton
    private static _instance : UserEventHandler;

    private _viewer : Viewer;
    private _ctrlPressed : boolean;
    private _wheelPressed : boolean;

    private constructor() {}

    public static getInstance() : UserEventHandler {
        if (!UserEventHandler._instance)
            UserEventHandler._instance = new UserEventHandler();
        return UserEventHandler._instance;
    }

    public initHandlers(viewer : Viewer){
        this._viewer = viewer;

        // LeftMouseButtonDown
        this._viewer.canvas.addEventListener('mousedown', (e : MouseEvent) =>{
            if (e.button == 0)
                console.log("leftMousePressed");
            if (e.button == 1)
                this._wheelPressed = true;
        });

        // LeftMouseButtonUp
        this._viewer.canvas.addEventListener('mouseup', (e : MouseEvent) => {
            if (e.button == 0)
                console.log("leftMouseUp");
            if (e.button == 1)
                this._wheelPressed = false;
        });

        // KeyDown
        window.addEventListener('keydown', (e : KeyboardEvent) => {
            if (e.key == "Control"){
                this._ctrlPressed = true;
            }
        })

        // KeyUp
        window.addEventListener('keyup', (e : KeyboardEvent) => {
            if (e.key == "Control"){
                this._ctrlPressed = false;
                
            }
        })

        //zoomIn/zoomOut
        this._viewer.canvas.addEventListener('wheel', (e : WheelEvent) =>{
            let delta : number = e.deltaY * 0.001;
            this._viewer.camera.moveForward(-delta);
        });
        
        

        this._viewer.canvas.addEventListener('mousemove', (e : MouseEvent) => {
            if (this._ctrlPressed || this._wheelPressed)
                this._viewer.camera.rotateCamera(e.movementY * 0.005, e.movementX * 0.005);
        })
        //....................................................
    }
}


export { UserEventHandler }