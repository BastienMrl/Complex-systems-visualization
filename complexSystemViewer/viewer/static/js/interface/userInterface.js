import { ViewerType } from "../viewerManager.js";
import { TransformType } from "../transformer/transformType.js";
import { TransformersInterface } from "./transformerInterface.js";
import { sendMessageToWorker, WorkerMessage } from "../workers/workerInterface.js";
import { SelectionManager, SelectionMode } from "./selectionTools/selectionManager.js";
import { AnimationInterface } from "./animation/animationInterface.js";
import { Stats } from "./stats.js";
export class UserInterface {
    // Singleton
    static _instance;
    _nbElements;
    _transformers;
    _animationCurves;
    _stats;
    _viewer;
    _selectionManager;
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
    set nbChannels(nb) {
        this._transformers.setNumberOfStatesOutput(nb);
    }
    get nbElements() {
        return this._nbElements;
    }
    initHandlers(viewer) {
        this._viewer = viewer;
        this._selectionManager = new SelectionManager(viewer);
        this.initMouseKeyHandlers();
        this.initInterfaceHandlers();
        this.initTransformers();
        this.initAnimationCurves();
        this.initStats();
    }
    initMouseKeyHandlers() {
        // LeftMouseButtonDown
        this._viewer.canvas.addEventListener('mousedown', (e) => {
            if (e.button == 1)
                this._wheelPressed = true;
        });
        // LeftMouseButtonUp
        this._viewer.canvas.addEventListener('mouseup', (e) => {
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
            this._viewer.onWheelMoved(delta);
        });
        this._viewer.canvas.addEventListener('mousemove', (e) => {
            if (this._ctrlPressed || this._wheelPressed)
                this._viewer.onMouseMoved(e.movementY * 0.005, e.movementX * 0.005);
        });
    }
    initInterfaceHandlers() {
        let playButton = document.querySelector('#buttonPlay');
        let pauseButton = document.querySelector('#buttonPause');
        let resetButton = document.querySelector('#buttonReset');
        let speedButton = document.querySelector('#buttonSpeed');
        let animationTimerEl = document.querySelector('#animationTimer');
        let visualizationPanel = document.getElementById("visualizationPanel");
        let simulationPanel = document.getElementById("simulationPanel");
        let foldVisualizationPanelButton = document.getElementById("foldVisualizationPanelButton");
        let foldSimulationPanelButton = document.getElementById("foldSimulationPanelButton");
        let toolButtons = document.getElementsByClassName("tool");
        let addTransformerButton = document.getElementById('buttonAddTransformer');
        let animableSelect = document.getElementById("animableSelect");
        let modelSelector = document.getElementById("modelSelector");
        let toolSettings = document.getElementById("toolSettings").children;
        let meshInputFile = document.getElementById("meshLoader");
        let viewerSelector = document.getElementById("currentViewer");
        playButton.addEventListener('click', () => {
            this._viewer.startVisualizationAnimation();
            pauseButton.classList.remove("active");
            playButton.classList.add("active");
            console.debug("START");
        });
        pauseButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation();
            playButton.classList.remove("active");
            pauseButton.classList.add("active");
            console.debug("STOP");
        });
        resetButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation();
            sendMessageToWorker(this._viewer.transmissionWorker, WorkerMessage.RESET);
            playButton.classList.remove("active");
            pauseButton.classList.add("active");
            console.debug("RESTART");
        });
        animationTimerEl.addEventListener('mouseleave', () => {
            let id = setTimeout(function () {
                animationTimerEl.style.display = 'none';
            }, 2000);
            animationTimerEl.onmouseenter = function () {
                clearTimeout(id);
            };
        });
        speedButton.addEventListener('click', () => {
            if (animationTimerEl.style.display == 'none')
                animationTimerEl.style.display = 'flex';
            else
                animationTimerEl.style.display = 'none';
        });
        foldVisualizationPanelButton.addEventListener("click", () => {
            visualizationPanel.classList.toggle("slideRight");
            foldVisualizationPanelButton.classList.toggle("slideRight");
        });
        foldSimulationPanelButton.addEventListener("click", () => {
            simulationPanel.classList.toggle("slideLeft");
            foldSimulationPanelButton.classList.toggle("slideLeft");
        });
        for (let i = 0; i < toolButtons.length; i++) {
            toolButtons.item(i).addEventListener("click", (e) => {
                let prevActiveTool = document.querySelectorAll(".toolActive:not(#tool" + toolButtons.item(i).id + ")");
                let sm = e.target.parentElement.getAttribute("selectionMode");
                switch (sm) {
                    case "BRUSH":
                        this._selectionManager.switchMode(SelectionMode.BRUSH);
                        break;
                    case "BOX":
                        this._selectionManager.switchMode(SelectionMode.BOX);
                        break;
                    default:
                        console.error("Selection mode " + sm + " is undefined.");
                }
                if (toolButtons.item(i).classList.contains("toolActive")) {
                    let toolSettings = document.getElementById("toolSettings");
                    toolSettings.style.alignItems = "center";
                    toolSettings.replaceChildren("");
                    let placehorlder = document.createElement("p");
                    placehorlder.innerText = "Select a tool";
                    toolSettings.appendChild(placehorlder);
                }
                else {
                    this.displayToolMenu();
                }
                toolButtons.item(i).classList.toggle("toolActive");
                if (prevActiveTool.length > 0) {
                    prevActiveTool[0].classList.remove("toolActive");
                }
            });
        }
        var nbAddedTransformer = 0;
        let superthis = this;
        addTransformerButton.addEventListener("click", (e) => {
            e.preventDefault();
            let transformertype = document.getElementById("transformerTypeSelector").value;
            let xhttp = new XMLHttpRequest();
            xhttp.open("GET", "addTransformer/" + transformertype, true);
            xhttp.onreadystatechange = function () {
                if (this.readyState == 4 && this.status == 200) {
                    let domParser = new DOMParser();
                    let newTransformer = domParser.parseFromString(this.responseText, "text/html").body.childNodes[0];
                    newTransformer.id = newTransformer.id + (nbAddedTransformer += 1);
                    let VP = document.getElementById("visualizationPanel");
                    VP.insertBefore(newTransformer, VP.lastChild.previousSibling);
                    superthis._transformers.addTransformerFromElement(newTransformer);
                }
            };
            xhttp.send();
        });
        animableSelect.addEventListener("change", () => {
            document.getElementsByClassName("afGridItem active")[0].classList.remove("active");
            let funcName = animableSelect.children[animableSelect.selectedIndex].getAttribute("animationFunction");
            document.getElementById(funcName).classList.add("active");
        });
        modelSelector.addEventListener("change", () => {
            sendMessageToWorker(this._viewer.transmissionWorker, WorkerMessage.CHANGE_SIMULATION, modelSelector.value);
            let xhttp = new XMLHttpRequest();
            xhttp.open("GET", "changeModel/" + modelSelector.value, true);
            xhttp.onreadystatechange = function () {
                if (this.readyState == 4 && this.status == 200) {
                    let domParser = new DOMParser();
                    let updateRules = domParser.parseFromString(this.responseText, "text/html").body.childNodes;
                    let prevRules = document.getElementById("rules");
                    let prevConfigSet = document.querySelectorAll(".configurationItem.simulationItem");
                    updateRules.forEach((elem) => {
                        prevRules.parentElement.appendChild(elem);
                    });
                    prevConfigSet.forEach((elem) => {
                        elem.remove();
                    });
                    superthis.initSimulationItem();
                }
            };
            xhttp.send();
        });
        this.initSimulationItem();
        // meshInputFile.addEventListener("change", () => {
        //     let meshFile : string = "/static/models/" + meshInputFile.value;
        //     this._viewer.currentMeshFile = meshFile;
        //     this._viewer.loadMesh(meshFile);
        // });
        viewerSelector.addEventListener("change", () => {
            let viewerType = null;
            switch (viewerSelector.value) {
                case ("Meshes"):
                    viewerType = ViewerType.MULTIPLE_MESHES;
                    break;
                case ("Texture"):
                    viewerType = ViewerType.TEXTURE;
                    break;
            }
            this._viewer.switchViewer(viewerType);
        });
        for (let i = 0; i < toolSettings.length; i++) {
            toolSettings.item(i).addEventListener("change", () => {
                this._selectionManager.setSelectionParameter(toolSettings.item(i).name, Number.parseFloat(toolSettings.item(i).value));
            });
        }
    }
    displayToolMenu() {
        let toolParam = JSON.parse(this._selectionManager.getSelectionParameter());
        let toolSettings = document.getElementById("toolSettings");
        toolSettings.replaceChildren("");
        if (!toolParam) {
            let nameElem = document.createElement("p");
            nameElem.textContent = "No parameters available";
            toolSettings.appendChild(nameElem);
            toolSettings.style.alignItems = "center";
            return;
        }
        toolSettings.style.alignItems = "initial";
        for (const toolName in toolParam) {
            let nameElem = document.createElement("h4");
            nameElem.textContent = toolName;
            let param = document.createElement("input");
            for (const paramSet in toolParam[toolName]) {
                param.setAttribute(paramSet, toolParam[toolName][paramSet]);
            }
            param.id = toolName;
            param.classList.add("toolParam");
            let valueDisplay = document.createElement("label");
            valueDisplay.setAttribute("for", toolName);
            valueDisplay.innerText = toolParam[toolName]["value"];
            param.addEventListener("change", () => {
                valueDisplay.innerText = param.value;
                this._selectionManager.setSelectionParameter(toolName, Number.parseFloat(param.value));
            });
            let paramContainer = document.createElement("div");
            paramContainer.classList.add("toolParamItem");
            paramContainer.appendChild(nameElem);
            paramContainer.appendChild(param);
            paramContainer.appendChild(valueDisplay);
            toolSettings.appendChild(paramContainer);
        }
    }
    initSimulationItem() {
        // ADD LISTENER FOR RULES ITEMS
        let rulesInputs = document.querySelectorAll("#rules .parameterItem input");
        let rulesInputsHandler = (e) => {
            sendMessageToWorker(this._viewer.transmissionWorker, WorkerMessage.UPDATE_RULES, this.parseInputToJson(e.target));
        };
        rulesInputs.forEach((input) => {
            input.addEventListener("change", rulesInputsHandler);
        });
        // ADD LISTENER FOR INIT_PARAMS
        let initParamInput = document.querySelectorAll("#initParam .parameterItem input");
        let initParamInputsHandler = (e) => {
            sendMessageToWorker(this._viewer.transmissionWorker, WorkerMessage.UPDATE_INIT_PARAM, this.parseInputToJson(e.target));
        };
        initParamInput.forEach((input) => {
            input.addEventListener("change", initParamInputsHandler);
        });
    }
    parseInputToJson(input) {
        let paramId = input.getAttribute("paramid");
        let paramIdSplited = paramId.split('_');
        let value;
        switch (input.type) {
            case "checkbox":
                value = input.checked;
                break;
            case "number":
                value = Number.parseFloat(input.value);
                break;
            default:
                value = input.value;
                break;
        }
        return JSON.stringify({
            "paramId": paramIdSplited[0],
            "subparam": paramIdSplited[1],
            "value": value
        });
    }
    initTransformers() {
        this._transformers = new TransformersInterface(this._viewer);
        document.querySelectorAll("[transformer]").forEach(e => {
            this._transformers.addTransformerFromElement(e);
        });
        this._transformers.updateProgram();
        let selector = document.getElementById("transformerTypeSelector");
        for (let type in TransformType) {
            let isValue = Number(type) >= 0;
            if (isValue) {
                let text = TransformType[type];
                let option = new Option(text, text);
                selector.add(option);
            }
        }
    }
    initAnimationCurves() {
        this._animationCurves = new AnimationInterface(this._viewer);
        let animationTimerEl = document.querySelector('#animationTimer');
        this._animationCurves.setDurationElement(animationTimerEl);
    }
    initStats() {
        let fpsElement = document.getElementById("renderingFps");
        let updtateElement = document.getElementById("updateMs");
        let renderingElement = document.getElementById("renderingMs");
        let pickingElement = document.getElementById("pickingMs");
        let totalElement = document.getElementById("totalMs");
        let transformationEl = document.getElementById("transformationTimer");
        let parsingEl = document.getElementById("parsingTimer");
        let receivingEl = document.getElementById("receivingTimer");
        let modelSelector = document.getElementById("modelSelector");
        this._stats = new Stats(fpsElement, updtateElement, renderingElement, pickingElement, totalElement, transformationEl, parsingEl, receivingEl);
        this._stats.withLog = true;
        this._viewer.stats = this._stats;
        this._selectionManager.stats = this._stats;
        this._stats.logModel(modelSelector.value);
        modelSelector.addEventListener("change", () => {
            this._stats.logModel(modelSelector.value);
        });
    }
}
