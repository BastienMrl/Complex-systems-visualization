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
        let GridSizeInput = (document.getElementById("gridSize") as HTMLInputElement);
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
        
        let foldButton = (document.getElementById("foldButton") as HTMLDivElement);
        
        let gridSizeInput = (document.getElementById("gridSize") as HTMLInputElement);

        let toolButtons = (document.getElementsByClassName("tool") as HTMLCollectionOf<HTMLDivElement>);


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
            document.getElementById("configurationPanel").classList.toggle("hidden")
            document.getElementById("foldButton").classList.toggle("hidden")
        });

        gridSizeInput.addEventListener("change", async () => {
            this._nbElements = (gridSizeInput.value as unknown as number)**2;
            this._viewer.initCurrentVisu(this._nbElements);
        });

        for(let i=0; i<toolButtons.length; i++){
            toolButtons.item(i).addEventListener("click", () => {
                let activeTool = document.getElementsByClassName("tool active")
                if(activeTool.length > 0){
                    activeTool[0].classList.remove("active");
                }
                toolButtons.item(i).classList.toggle("active");
                if (i == 0) {
                    this._viewer.usePicking = !this._viewer.usePicking;
                }
            });
        }

    }

    private initTransformers(){
        this._transformers = new TransformersInterface(this._viewer);

        let colorTransformerElement = document.getElementById("2") as HTMLElement;
        this._transformers.addTransformerFromElement(colorTransformerElement);
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
        this._currentStatesTransformer.addTransformer(TransformType.POSITION_X, InputType.POSITION_X, [1.95]);
        this._currentStatesTransformer.addTransformer(TransformType.POSITION_Z, InputType.POSITION_Y, [1.95]);
        this._currentStatesTransformer.addTransformer(TransformType.POSITION_Y, InputType.STATE_0, [1.5]);
    }

    public addTransformerFromElement(element : HTMLElement){
        const transformType = this.getTransformType(element);
        const inputType = this.getInputType(element);
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

        // TODO: add functions to disconnect / delete transformer
    }

    public updateProgram(){
        this._viewer.updateProgamsTransformers(this._currentStatesTransformer);
    }

    // TODO: return value according to HTMLElement
    private getTransformType(element : HTMLElement){        
        return TransformType.COLOR;
    }

    // TODO: return value accroding to HTMLElement
    private getInputType(element : HTMLElement){
        return InputType.STATE_0;
    }

    // TODO : fill with right ids
    private getParamsElements(element : HTMLElement){
        switch (this.getTransformType(element)){
            case TransformType.COLOR:
                let colorAliveInput = (document.getElementById("aliveColor") as HTMLInputElement);
                let colorDeadInput = (document.getElementById("deadColor") as HTMLInputElement);
                return [colorDeadInput, colorAliveInput];
            case TransformType.COLOR_R:
            case TransformType.COLOR_G:
            case TransformType.COLOR_B:
            case TransformType.POSITION_X:
            case TransformType.POSITION_Y:
            case TransformType.POSITION_Z:
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
