import { SocketHandler } from "./socketHandler.js";
import { Viewer } from "./viewer.js";

export class UserInterface {
    // Singleton
    private static _instance : UserInterface;

    private _nbInstances : number;

    private _viewer : Viewer;
    private _socketHandler : SocketHandler;

    private _ctrlPressed : boolean;
    private _wheelPressed : boolean;

    private constructor() {
        this._socketHandler = SocketHandler.getInstance();
        this._nbInstances = 10 * 10;
    }

    public static getInstance() : UserInterface {
        if (!UserInterface._instance)
            UserInterface._instance = new UserInterface();
        return UserInterface._instance;
    }

    public initHandlers(viewer : Viewer){
        this._viewer = viewer;
        this.initMouseKeyHandlers();
        this.initInterfaceHandlers();
    }

    private initMouseKeyHandlers(){
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
            if (e.key == "Shift"){
                this._ctrlPressed = true;
            }
        });

        // KeyUp
        window.addEventListener('keyup', (e : KeyboardEvent) => {
            if (e.key == "Shift"){
                this._ctrlPressed = false;
                
            }
        });

        //zoomIn/zoomOut
        this._viewer.canvas.addEventListener('wheel', (e : WheelEvent) =>{
            let delta : number = e.deltaY * 0.001;
            this._viewer.camera.moveForward(-delta);
        });
        
        

        this._viewer.canvas.addEventListener('mousemove', (e : MouseEvent) => {
            if (this._ctrlPressed || this._wheelPressed)
                this._viewer.camera.rotateCamera(e.movementY * 0.005, e.movementX * 0.005);
        });
    }

    private initInterfaceHandlers()
    {
        let playButton = (document.querySelector('#buttonPlay') as HTMLButtonElement);
        let pauseButton = (document.querySelector('#buttonPause') as HTMLButtonElement);
        let restartButton = (document.querySelector('#buttonRestart') as HTMLButtonElement);
        let foldButton = (document.getElementById("foldButton") as HTMLDivElement);


        playButton.addEventListener('click', (e : MouseEvent) => {
            if (!this._socketHandler.isRunning){
                this._socketHandler.start(this._nbInstances);
                console.log("START");
            }
        });

        pauseButton.addEventListener('click', (e : MouseEvent) => {
            if (this._socketHandler.isRunning){
                this._socketHandler.stop();
                console.log("STOP");
            }
        });

        restartButton.addEventListener('click', (e : MouseEvent) => {
            if (this._socketHandler.isRunning)
                this._socketHandler.stop();
            this._viewer.initCurrentVisu(this._nbInstances);
            console.log("RESTART");
        })

        foldButton.addEventListener("click", () => {
            document.getElementById("configurationPanel").classList.toggle("hidden")
            document.getElementById("foldButton").classList.toggle("hidden")
        });
    }


    public get nbInstances() : number {
        return this._nbInstances;
    }
}
