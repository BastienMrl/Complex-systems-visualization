import { SocketHandler } from "./socketHandler.js";
import { idColor, transformer } from "./index.js";
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
        let GridSizeInput = document.getElementById("gridSize");
        this._nbInstances = GridSizeInput.value ** 2;
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
    hexToRgbA(hex) {
        var c;
        if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
            c = hex.substring(1).split('');
            if (c.length == 3) {
                c = [c[0], c[0], c[1], c[1], c[2], c[2]];
            }
            c = '0x' + c.join('');
            return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
        }
        throw new Error('Bad Hex');
    }
    initInterfaceHandlers() {
        let playButton = document.querySelector('#buttonPlay');
        let pauseButton = document.querySelector('#buttonPause');
        let restartButton = document.querySelector('#buttonRestart');
        let foldButton = document.getElementById("foldButton");
        let colorAliveInput = document.getElementById("aliveColor");
        let colorDeadInput = document.getElementById("deadColor");
        let gridSizeInput = document.getElementById("gridSize");
        let toolButtons = document.getElementsByClassName("tool");
        playButton.addEventListener('click', () => {
            if (!this._socketHandler.isRunning) {
                this._socketHandler.start(this._nbInstances);
                console.log("START");
            }
        });
        pauseButton.addEventListener('click', () => {
            if (this._socketHandler.isRunning) {
                this._socketHandler.stop();
                console.log("STOP");
            }
        });
        restartButton.addEventListener('click', () => {
            if (this._socketHandler.isRunning)
                this._socketHandler.stop();
            this._viewer.initCurrentVisu(this._nbInstances);
            console.log("RESTART");
        });
        foldButton.addEventListener("click", () => {
            document.getElementById("configurationPanel").classList.toggle("hidden");
            document.getElementById("foldButton").classList.toggle("hidden");
        });
        colorAliveInput.addEventListener("input", () => {
            let color = this.hexToRgbA(colorAliveInput.value);
            transformer.setParams(idColor, null, color);
        });
        colorDeadInput.addEventListener("input", () => {
            let color = this.hexToRgbA(colorDeadInput.value);
            transformer.setParams(idColor, color, null);
        });
        gridSizeInput.addEventListener("input", async () => {
            this._nbInstances = gridSizeInput.value ** 2;
            this._viewer.initCurrentVisu(this._nbInstances);
        });
        for (let i = 0; i < toolButtons.length; i++) {
            toolButtons.item(i).addEventListener("click", () => {
                let activeTool = document.getElementsByClassName("tool active");
                if (activeTool.length > 0) {
                    activeTool[0].classList.remove("active");
                }
                toolButtons.item(i).classList.toggle("active");
            });
        }
    }
    get nbInstances() {
        return this._nbInstances;
    }
}
