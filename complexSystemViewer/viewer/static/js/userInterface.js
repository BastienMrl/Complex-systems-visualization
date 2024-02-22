import { AnimableValue } from "./viewer.js";
import { InputType, StatesTransformer, TransformType } from "./statesTransformer.js";
const MAX_REFRESH_RATE = 20.;
const MIN_REFRESH_RATE = 0.5;
const REFRESH_STEP = 0.5;
const DEFAULT_REFRESH_RATE = 6.;
export class UserInterface {
    // Singleton
    static _instance;
    _nbElements;
    _transformers;
    _animationCurves;
    _viewer;
    _ctrlPressed;
    _wheelPressed;
    constructor() {
        let GridSizeInput = document.querySelector("input[paramId=gridSize]");
        this._nbElements = GridSizeInput.value ** 2;
    }
    static getInstance() {
        if (!UserInterface._instance)
            UserInterface._instance = new UserInterface();
        return UserInterface._instance;
    }
    get nbElements() {
        return this._nbElements;
    }
    initHandlers(viewer) {
        this._viewer = viewer;
        this.initMouseKeyHandlers();
        this.initInterfaceHandlers();
        this.initTransformers();
        this.initAnimationCurves();
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
        let restartButton = document.querySelector('#buttonRestart');
        let timerButton = document.querySelector('#buttonTimer');
        let animationTimerEl = document.querySelector('#animationTimer');
        let configurationPanel = document.getElementById("configurationPanel");
        let foldButton = document.getElementById("foldButton");
        let gridSizeInput = document.querySelector("input[paramId=gridSize]");
        let toolButtons = document.getElementsByClassName("tool");
        let addTransformerButton = document.querySelector('#buttonAddTransformer');
        playButton.addEventListener('click', () => {
            this._viewer.startVisualizationAnimation();
            console.log("START");
        });
        pauseButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation();
            console.log("STOP");
        });
        restartButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation();
            this._viewer.initCurrentVisu(this._nbElements);
            console.log("RESTART");
        });
        animationTimerEl.addEventListener('mouseleave', () => {
            let id = setTimeout(function () {
                animationTimerEl.style.display = 'none';
            }, 2000);
            animationTimerEl.onmouseenter = function () {
                clearTimeout(id);
            };
        });
        timerButton.addEventListener('click', () => {
            if (animationTimerEl.style.display == 'none')
                animationTimerEl.style.display = 'flex';
            else
                animationTimerEl.style.display = 'none';
        });
        foldButton.addEventListener("click", () => {
            configurationPanel.classList.toggle("hidden");
            foldButton.classList.toggle("hidden");
        });
        gridSizeInput.addEventListener("change", async () => {
            this._nbElements = gridSizeInput.value ** 2;
            this._viewer.initCurrentVisu(this._nbElements);
        });
        for (let i = 0; i < toolButtons.length; i++) {
            toolButtons.item(i).addEventListener("click", () => {
                let activeTool = document.getElementsByClassName("tool active");
                if (activeTool.length > 0) {
                    activeTool[0].classList.remove("active");
                }
                toolButtons.item(i).classList.toggle("active");
                if (i == 0) {
                    this._viewer.usePicking = !this._viewer.usePicking;
                }
            });
        }
        var nbAddedTransformer = 0;
        let superthis = this;
        addTransformerButton.addEventListener("click", (e) => {
            e.preventDefault();
            let transformertype = document.getElementById("transformerTypeSelector").value;
            let selectedModel = document.getElementById("modelSelector").value;
            let xhttp = new XMLHttpRequest();
            xhttp.open("GET", "addTranformerURL/" + selectedModel + "/" + transformertype, true);
            xhttp.onreadystatechange = function () {
                if (this.readyState == 4 && this.status == 200) {
                    let domParser = new DOMParser();
                    let newTransformer = domParser.parseFromString(this.responseText, "text/html").body.childNodes[0];
                    newTransformer.id = newTransformer.id + (nbAddedTransformer += 1);
                    let CP = document.getElementById("configurationPanel");
                    CP.insertBefore(newTransformer, CP.lastChild.previousSibling);
                    superthis._transformers.addTransformerFromElement(newTransformer);
                }
            };
            xhttp.send();
        });
    }
    initTransformers() {
        this._transformers = new TransformersInterface(this._viewer);
        let colorTransformerElement = document.getElementById("colorTransformer");
        let positionXElement = document.getElementById("positionX");
        let positionYElement = document.getElementById("positionY");
        let positionZElement = document.getElementById("positionZ");
        this._transformers.addTransformerFromElement(colorTransformerElement);
        this._transformers.addTransformerFromElement(positionXElement);
        this._transformers.addTransformerFromElement(positionYElement);
        this._transformers.addTransformerFromElement(positionZElement);
        this._transformers.updateProgram();
    }
    initAnimationCurves() {
        this._animationCurves = new AnimationInterface(this._viewer);
        let animationTimerEl = document.querySelector('#animationTimer');
        this._animationCurves.setDurationElement(animationTimerEl);
    }
}
export class TransformersInterface {
    _viewer;
    _currentStatesTransformer;
    constructor(viewer) {
        this._viewer = viewer;
        this._currentStatesTransformer = new StatesTransformer();
    }
    addTransformerFromElement(element) {
        const inputElement = this.getInputTypeElement(element);
        const inputType = this.getInputType(inputElement);
        const transformType = this.getTransformType(element);
        const paramsElements = this.getParamsElements(element);
        let params = [];
        paramsElements.forEach(e => {
            params.push(e.value);
        });
        const id = this._currentStatesTransformer.addTransformer(transformType, inputType, params);
        paramsElements.forEach((e, i) => {
            e.addEventListener("input", () => {
                let newParams = new Array(params.length).fill(null);
                newParams[i] = e.value;
                this._currentStatesTransformer.setParams(id, newParams);
                this.updateProgram();
            });
        });
        inputElement.addEventListener("change", () => {
            this._currentStatesTransformer.setInputType(id, this.getInputType(inputElement));
            this.updateProgram();
        });
        // TODO: add functions to disconnect / delete transformer
    }
    updateProgram() {
        this._viewer.updateProgamsTransformers(this._currentStatesTransformer);
    }
    // TODO: return value according to HTMLElement
    getTransformType(element) {
        switch (element.getAttribute("transformer")) {
            case "COLOR":
                return TransformType.COLOR;
            case "COLOR_R":
                return TransformType.COLOR_R;
            case "COLOR_G":
                return TransformType.COLOR_G;
            case "COLOR_B":
                return TransformType.COLOR_B;
            case "POSITION_X":
                return TransformType.POSITION_X;
            case "POSITION_Y":
                return TransformType.POSITION_Y;
            case "POSITION_Z":
                return TransformType.POSITION_Z;
        }
    }
    // TODO: return value accroding to HTMLElement
    getInputTypeElement(parent) {
        return parent.getElementsByClassName("visualizationInput")[0].children[0];
    }
    getInputType(element) {
        switch (element.value) {
            case "POSITION_X":
                return InputType.POSITION_X;
            case "POSITION_Y":
                return InputType.POSITION_Y;
            case "POSITION_Z":
                return InputType.POSITION_Z;
            case "STATE_0":
                return InputType.STATE_0;
        }
    }
    // TODO : fill with right ids
    getParamsElements(parent) {
        switch (this.getTransformType(parent)) {
            case TransformType.COLOR:
                let colorAliveInput = parent.querySelector("input[paramId=c1]");
                let colorDeadInput = parent.querySelector("input[paramId=c0]");
                return [colorDeadInput, colorAliveInput];
            case TransformType.COLOR_R:
            case TransformType.COLOR_G:
            case TransformType.COLOR_B:
            case TransformType.POSITION_X:
                return [parent.querySelector("input[paramId=factor]")];
            case TransformType.POSITION_Y:
                return [parent.querySelector("input[paramId=factor]")];
            case TransformType.POSITION_Z:
                return [parent.querySelector("input[paramId=factor]")];
        }
    }
}
class AnimationInterface {
    _viewer;
    constructor(viewer) {
        this._viewer = viewer;
        //.... AnimationCurves ....
        // default animation curve is linear
        // ease out expo from https://easings.net/
        let easeOut = function (time) { return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
        let fc0 = function (time) { return 1; };
        this._viewer.bindAnimationCurve(AnimableValue.COLOR, easeOut);
        this._viewer.bindAnimationCurve(AnimableValue.TRANSLATION, easeOut);
        //.........................
    }
    setDurationElement(element) {
        let input = document.getElementById("inputTimer");
        let label = document.getElementById("labelTimer");
        input.min = `${MIN_REFRESH_RATE}`;
        input.max = `${MAX_REFRESH_RATE}`;
        input.step = `${REFRESH_STEP}`;
        input.value = `${DEFAULT_REFRESH_RATE}`;
        label.innerHTML = `<strong>${input.value}</strong> steps per second`;
        this._viewer.animationDuration = (1. / Number(input.value));
        input.addEventListener("input", () => {
            label.innerHTML = `<strong>${input.value}</strong> steps per second`;
            this._viewer.animationDuration = (1. / Number(input.value));
        });
        element.style.display = 'none';
    }
}
