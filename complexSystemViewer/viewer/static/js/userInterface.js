import { SocketHandler } from "./socketHandler.js";
export class UserInterface {
    // Singleton
    static _instance;
    _nbInstances;
    _viewer;
    _socketHandler;
    _ctrlPressed;
    _wheelPressed;
    constructor() {
        this._socketHandler = SocketHandler.getInstance();
        this._nbInstances = 200 * 200;
    }
    static getInstance() {
        if (!UserInterface._instance)
            UserInterface._instance = new UserInterface();
        return UserInterface._instance;
    }
    initHandlers(viewer) {
        this._viewer = viewer;
        this.initMouseKeyHandlers();
        this.initInterfaceHandlers();
    }
    initMouseKeyHandlers() {
        // LeftMouseButtonDown
        this._viewer.canvas.addEventListener('mousedown', (e) => {
            if (e.button == 0)
                console.log("leftMousePressed");
            if (e.button == 1)
                this._wheelPressed = true;
        });
        // LeftMouseButtonUp
        this._viewer.canvas.addEventListener('mouseup', (e) => {
            if (e.button == 0)
                console.log("leftMouseUp");
            if (e.button == 1)
                this._wheelPressed = false;
        });
        // KeyDown
        window.addEventListener('keydown', (e) => {
            if (e.key == "Shift") {
                this._ctrlPressed = true;
            }
        });
        // KeyUp
        window.addEventListener('keyup', (e) => {
            if (e.key == "Shift") {
                this._ctrlPressed = false;
            }
        });
        //zoomIn/zoomOut
        this._viewer.canvas.addEventListener('wheel', (e) => {
            let delta = e.deltaY * 0.001;
            this._viewer.camera.moveForward(-delta);
        });
        this._viewer.canvas.addEventListener('mousemove', (e) => {
            if (this._ctrlPressed || this._wheelPressed)
                this._viewer.camera.rotateCamera(e.movementY * 0.005, e.movementX * 0.005);
        });
    }
    initInterfaceHandlers() {
        let playButton = document.querySelector('#buttonPlay');
        let pauseButton = document.querySelector('#buttonPause');
        playButton.addEventListener('click', (e) => {
            console.log(this._socketHandler);
            if (!this._socketHandler.isRunning) {
                this._socketHandler.start(this._nbInstances);
                console.log("START");
            }
        });
        pauseButton.addEventListener('click', (e) => {
            if (this._socketHandler.isRunning) {
                this._socketHandler.stop();
                console.log(this._socketHandler);
            }
        });
    }
    get nbInstances() {
        return this._nbInstances;
    }
}
