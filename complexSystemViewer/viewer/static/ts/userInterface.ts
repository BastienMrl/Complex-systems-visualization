import { AnimableValue, Viewer } from "./viewer.js";
import { InputType, StatesTransformer, TransformType } from "./statesTransformer.js";

const MAX_REFRESH_RATE = 20.;
const MIN_REFRESH_RATE = 0.5;
const REFRESH_STEP = 0.5;
const DEFAULT_REFRESH_RATE = 6.;

export class UserInterface {
    // Singleton
    private static _instance : UserInterface;

    private _nbElements : number;

    private _transformers : TransformersInterface;
    private _animationCurves : AnimationInterface;

    private _viewer : Viewer;

    private _ctrlPressed : boolean;
    private _wheelPressed : boolean;

    private constructor() {
        let GridSizeInput = (document.querySelector("input[paramId=gridSize]") as HTMLInputElement);
        this._nbElements = (GridSizeInput.value as unknown as number) ** 2;
    }

    public static getInstance() : UserInterface {
        if (!UserInterface._instance)
            UserInterface._instance = new UserInterface();
        return UserInterface._instance;
    }

    public get nbElements() : number {
        return this._nbElements;
    }

    public initHandlers(viewer : Viewer){
        this._viewer = viewer;
        this.initMouseKeyHandlers();
        this.initInterfaceHandlers();
        this.initTransformers();
        this.initAnimationCurves();
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
        let timerButton = (document.querySelector('#buttonTimer') as HTMLButtonElement);
        
        let animationTimerEl = (document.querySelector('#animationTimer') as HTMLDivElement);
        
        let configurationPanel = (document.getElementById("configurationPanel") as HTMLDivElement);

        let foldButton = (document.getElementById("foldButton") as HTMLDivElement);
        
        let gridSizeInput = (document.querySelector("input[paramId=gridSize]") as HTMLInputElement);

        let toolButtons = (document.getElementsByClassName("tool") as HTMLCollectionOf<HTMLDivElement>);

        let addTransformerButton = (document.querySelector('#buttonAddTransformer') as HTMLButtonElement);


        playButton.addEventListener('click', () => {
            this._viewer.startVisualizationAnimation();
            console.log("START");
        });

        pauseButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation()
            console.log("STOP");
        });

        restartButton.addEventListener('click', () => {
            this._viewer.stopVisualizationAnimation();
            this._viewer.initCurrentVisu(this._nbElements);
            console.log("RESTART");
        });

        animationTimerEl.addEventListener('mouseleave', () => {
            let id = setTimeout(function() {
                animationTimerEl.style.display = 'none';
            }, 2000);
            animationTimerEl.onmouseenter = function(){
                clearTimeout(id);
            }
        });


        timerButton.addEventListener('click', () => {
            if (animationTimerEl.style.display == 'none')
                animationTimerEl.style.display = 'flex';
            else
                animationTimerEl.style.display = 'none';
        });

        foldButton.addEventListener("click", () => {
            configurationPanel.classList.toggle("hidden")
            foldButton.classList.toggle("hidden")
        });

        gridSizeInput.addEventListener("change", async () => {
            this._nbElements = (gridSizeInput.value as unknown as number)**2;
            this._viewer.initCurrentVisu(this._nbElements);
        });

        for(let i=0; i<toolButtons.length; i++){
            toolButtons.item(i).addEventListener("click", () => {
                let prevActiveTool = document.querySelectorAll(".toolActive:not(#tool" + toolButtons.item(i).id +")");
                if (i == 0 || prevActiveTool[0].id == "tool1") {
                    this._viewer.usePicking = !this._viewer.usePicking;
                }
                toolButtons.item(i).classList.toggle("toolActive");
                if(prevActiveTool.length > 0){
                    prevActiveTool[0].classList.remove("toolActive");
                }
            });
        }

        var nbAddedTransformer = 0;
        let superthis = this;
        addTransformerButton.addEventListener("click", (e) => {
            e.preventDefault();
            let transformertype = (document.getElementById("transformerTypeSelector") as HTMLSelectElement).value
            let selectedModel = (document.getElementById("modelSelector") as HTMLSelectElement).value
            let xhttp = new XMLHttpRequest()
            xhttp.open("GET", "addTranformerURL/" + selectedModel + "/" + transformertype, true);
            xhttp.onreadystatechange = function() {
                if(this.readyState == 4 && this.status == 200){
                    let domParser = new DOMParser();
                    let newTransformer = domParser.parseFromString(this.responseText, "text/html").body.childNodes[0] as HTMLDivElement;
                    newTransformer.id = newTransformer.id + (nbAddedTransformer+=1) 
                    let CP = document.getElementById("configurationPanel");
                    CP.insertBefore(newTransformer, CP.lastChild.previousSibling);
                    superthis._transformers.addTransformerFromElement(newTransformer)
                }
            }
            xhttp.send();
        });

    }

    private initTransformers(){
        this._transformers = new TransformersInterface(this._viewer);

        let colorTransformerElement = document.getElementById("colorTransformer") as HTMLElement;
        let positionXElement = document.getElementById("positionX") as HTMLElement;
        let positionYElement = document.getElementById("positionY") as HTMLElement;
        let positionZElement = document.getElementById("positionZ") as HTMLElement;
        // let colorRElement = document.getElementById("colorR") as HTMLElement;
        // let colorGElement = document.getElementById("colorG") as HTMLElement;
        // let colorBElement = document.getElementById("colorB") as HTMLElement;


        this._transformers.addTransformerFromElement(colorTransformerElement);
        this._transformers.addTransformerFromElement(positionXElement);
        this._transformers.addTransformerFromElement(positionYElement);
        this._transformers.addTransformerFromElement(positionZElement);
        // this._transformers.addTransformerFromElement(colorRElement);
        // this._transformers.addTransformerFromElement(colorGElement);
        // this._transformers.addTransformerFromElement(colorBElement);

        this._transformers.updateProgram();
    }

    private initAnimationCurves(){
        this._animationCurves = new AnimationInterface(this._viewer);
        let animationTimerEl = (document.querySelector('#animationTimer') as HTMLDivElement);
        this._animationCurves.setDurationElement(animationTimerEl);
    }

}

export class TransformersInterface {
    private _viewer : Viewer;
    private _currentStatesTransformer : StatesTransformer;

    public constructor(viewer : Viewer){
        this._viewer = viewer;
        this._currentStatesTransformer = new StatesTransformer();
        this._viewer.pickingTool.setTransformer(this._currentStatesTransformer);
    }

    public addTransformerFromElement(element : HTMLElement){
        const inputElement = this.getInputTypeElement(element);
        const inputType = this.getInputType(inputElement);

        const deleteButton = (element.getElementsByClassName("deleteButton")[0] as HTMLButtonElement);
        
        const transformType = this.getTransformType(element);
        const paramsElements = this.getParamsElements(element);
        console.log(paramsElements)
        let params = [];
        paramsElements.forEach(e => {
            params.push(e.value);
        });
        const id = this._currentStatesTransformer.addTransformer(transformType, inputType, params);
        paramsElements.forEach((e, i) => {
            e.addEventListener("change", () => {
                let newParams = new Array(params.length).fill(null);
                newParams[i] = e.value;
                this._currentStatesTransformer.setParams(id, newParams);
                this.updateProgram();
            });
            e.dispatchEvent(new Event('change'));
        });
        
        
        inputElement.addEventListener("change", () => {
            this._currentStatesTransformer.setInputType(id, this.getInputType(inputElement));
            this.updateProgram();
        });

        //function to disconnect / delete transformer
        if(deleteButton){
            deleteButton.addEventListener("click", () => {
                this._currentStatesTransformer.removeTransformer(id);
                deleteButton.parentElement.remove();
                this.updateProgram();
                console.log("deleted");
            });
        }
    }

    public updateProgram(){
        this._viewer.updateProgamsTransformers(this._currentStatesTransformer);
    }

    // TODO: return value according to HTMLElement
    private getTransformType(element : HTMLElement) : TransformType{        
        switch(element.getAttribute("transformer")){
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
    private getInputTypeElement(parent : HTMLElement) : HTMLSelectElement{
        return parent.getElementsByClassName("visualizationInput")[0].children[0] as HTMLSelectElement;
    }

    private getInputType(element : HTMLSelectElement) : InputType{
        switch (element.value) {
            case "POSITION_X" : 
                return InputType.POSITION_X;
            case "POSITION_Y" :
                return InputType.POSITION_Y;
            case "POSITION_Z" : 
                return InputType.POSITION_Z;
            case "STATE_0":
                return InputType.STATE_0;
        }
    }

    // TODO : fill with right ids
    private getParamsElements(parent : HTMLElement){
        switch (this.getTransformType(parent)){
            case TransformType.COLOR:
                let colorAliveInput = parent.querySelector("input[paramId=c1]") as HTMLInputElement;
                let colorDeadInput = parent.querySelector("input[paramId=c0]") as HTMLInputElement;     
                return [colorDeadInput, colorAliveInput];
            case TransformType.COLOR_R :
            case TransformType.COLOR_G :
            case TransformType.COLOR_B :
                let min = parent.querySelector("input[paramId=rangeMin]") as HTMLInputElement;
                let max = parent.querySelector("input[paramId=rangeMax]") as HTMLInputElement;
                return [min, max];
            case TransformType.POSITION_X :
            case TransformType.POSITION_Y :
            case TransformType.POSITION_Z :
                return [parent.querySelector("input[paramId=factor]") as HTMLInputElement];
        }
    }
}


class AnimationInterface{
    private _viewer : Viewer;

    constructor(viewer : Viewer){
        this._viewer = viewer;
        //.... AnimationCurves ....
        // default animation curve is linear
        
        // ease out expo from https://easings.net/
        let easeOut = function(time : number){ return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
        let fc0 = function(time : number){ return 1 };
        this._viewer.bindAnimationCurve(AnimableValue.COLOR, easeOut);
        this._viewer.bindAnimationCurve(AnimableValue.TRANSLATION, easeOut);
        
        
        //.........................
    }

    public setDurationElement(element : HTMLElement){
        let input = document.getElementById("inputTimer") as HTMLInputElement;
        let label = document.getElementById("labelTimer") as HTMLLabelElement;
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
