import { AnimableValue, Viewer } from "./viewer.js";
import { InputType, TransformerBuilder, TransformType } from "./transformerBuilder.js";
import { sendMessageToWorker, WorkerMessage } from "./workers/workerInterface.js"
import { SelectionMode } from "./selectionTools/selectionManager.js";

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
            if (e.button == 1)
                this._wheelPressed = true;
        });

        // LeftMouseButtonUp
        this._viewer.canvas.addEventListener('mouseup', (e : MouseEvent) => {
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
        
        let visualizationPanel = (document.getElementById("visualizationPanel") as HTMLDivElement);
        let simulationPanel = (document.getElementById("simulationPanel") as HTMLDivElement);

        let foldVisualizationPanelButton = (document.getElementById("foldVisualizationPanelButton") as HTMLDivElement);
        let foldSimulationPanelButton = (document.getElementById("foldSimulationPanelButton") as HTMLDivElement);

        let toolButtons = (document.getElementsByClassName("tool") as HTMLCollectionOf<HTMLDivElement>);

        let addTransformerButton = (document.getElementById('buttonAddTransformer') as HTMLButtonElement);

        let animableSelect = (document.getElementById("animableSelect") as HTMLSelectElement);

        let modelSelector = (document.getElementById("modelSelector") as HTMLSelectElement);

        let gridSizeInput = (document.querySelector("input[paramid=\"gridSize\"]") as HTMLInputElement);

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

        foldVisualizationPanelButton.addEventListener("click", () => {
            visualizationPanel.classList.toggle("slideRight")
            foldVisualizationPanelButton.classList.toggle("slideRight")
        });

        foldSimulationPanelButton.addEventListener("click", () => {
            simulationPanel.classList.toggle("slideLeft")
            foldSimulationPanelButton.classList.toggle("slideLeft")
        });

        gridSizeInput.addEventListener("change", async () => {
            this._nbElements = (gridSizeInput.value as unknown as number)**2;
            this._viewer.initCurrentVisu(this._nbElements);
        });

        for(let i=0; i<toolButtons.length; i++){
            toolButtons.item(i).addEventListener("click", () => {
                let prevActiveTool = document.querySelectorAll(".toolActive:not(#tool" + toolButtons.item(i).id +")");
                if (i == 0) {
                    this._viewer.selectionManager.switchMode(SelectionMode.BRUSH);
                }
                if (i == 1){
                    this._viewer.selectionManager.switchMode(SelectionMode.BOX);
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
            let xhttp = new XMLHttpRequest()
            xhttp.open("GET", "addTranformerURL/" + modelSelector.value + "/" + transformertype, true);
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

        animableSelect.addEventListener("change", () => {
            document.getElementsByClassName("afGridItem active")[0].classList.remove("active")
            let funcName = animableSelect.children[animableSelect.selectedIndex].getAttribute("animationFunction");
            document.getElementById(funcName).classList.add("active");
        });

        modelSelector.addEventListener("change", () => {
            let xhttp = new XMLHttpRequest()
            xhttp.open("GET", "changeModel/" + modelSelector.value, true);
            xhttp.onreadystatechange = function() {
                if(this.readyState == 4 && this.status == 200){
                    let domParser = new DOMParser();
                    let updateRules = domParser.parseFromString(this.responseText, "text/html").body.childNodes[0] as HTMLDivElement;
                    let prevRules = document.getElementById("rules") as HTMLDivElement;
                    prevRules.parentElement.appendChild(updateRules)
                    prevRules.parentElement.removeChild(prevRules)
                    superthis.initRulesListener()
                }
            }
            xhttp.send();
        })

        this.initRulesListener()

    }

    private initRulesListener(){
        let paramsInputRule = document.querySelectorAll("#rules .parameterItem input");
        let sendSimuRules = (e) =>{
            let input = (e.target as HTMLInputElement)
            let paramId = input.getAttribute("paramid");
            let paramIdSplited = paramId.split('_')
            let json = JSON.stringify({
                "paramId":paramIdSplited[0],
                "subparam":paramIdSplited[1],
                "value": Number.parseFloat(input.value)
            })
            sendMessageToWorker(this._viewer.transmissionWorker, WorkerMessage.UPDATE_RULES, json);
        }

        paramsInputRule.forEach( (input)=>{
            input.addEventListener("change", sendSimuRules)
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
    private _currentTransformerBuilder : TransformerBuilder;

    public constructor(viewer : Viewer){
        this._viewer = viewer;
        this._currentTransformerBuilder = new TransformerBuilder();
        this._viewer.selectionManager.setTransformer(this._currentTransformerBuilder);
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
        const id = this._currentTransformerBuilder.addTransformer(transformType, inputType, params);
        paramsElements.forEach((e, i) => {
            e.addEventListener("change", () => {
                let newParams = new Array(params.length).fill(null);
                newParams[i] = e.value;
                this._currentTransformerBuilder.setParams(id, newParams);
                this.updateProgram();
            });
            e.dispatchEvent(new Event('change'));
        });
        
        
        inputElement.addEventListener("change", () => {
            this._currentTransformerBuilder.setInputType(id, this.getInputType(inputElement));
            this.updateProgram();
        });

        //function to disconnect / delete transformer
        if(deleteButton){
            deleteButton.addEventListener("click", () => {
                this._currentTransformerBuilder.removeTransformer(id);
                deleteButton.parentElement.remove();
                this.updateProgram();
                console.log("deleted");
            });
        }
    }

    public updateProgram(){
        this._viewer.updateProgamsTransformers(this._currentTransformerBuilder);
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

// easeing functions from https://easings.net/
class AnimationFuction{
    static easeOut = function(time : number){ return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
    static easeOutElastic = function(time: number): number {
            const c4 = (2 * Math.PI) / 3;
            return time === 0 ? 0 : time === 1 ? 1 : Math.pow(2, -10 * time) * Math.sin((time * 10 - 0.75) * c4) + 1;
    }
    static easeInBack = function(time: number): number {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return c3 * time * time * time - c1 * time * time;
    }
    static fc0 = function(time : number){ return time < 0.5 ? 0 : 1 };
    static linear = function(time : number){return time};
    static easeInExpo =function(time: number){
        return time === 0 ? 0 : Math.pow(2, 10 * time - 10);
    }
    static easeInOutBack = function(time: number){
    const c1 = 1.70158;
    const c2 = c1 * 1.525;

    return time < 0.5
    ? (Math.pow(2 * time, 2) * ((c2 + 1) * 2 *  time - c2)) / 2
    : (Math.pow(2 * time - 2, 2) * ((c2 + 1) * (time * 2 - 2) + c2) + 2) / 2;
    };   

    static retrieveFunction(functionName:string){
        switch (functionName) {
            case "easeOut":
                return AnimationFuction.easeOut
            case "easeOutElastic":
                return AnimationFuction.easeOutElastic
            case "fc0":
                return AnimationFuction.fc0
            case "easeInBack":
                return AnimationFuction.easeInBack;
            case "linear":
                return AnimationFuction.linear;
            case "easeInExpo":
                return AnimationFuction.easeInExpo;
            case "easeInOutBack":
                return AnimationFuction.easeInOutBack;
            default:
                break;
        }    
    }
}

class AnimationInterface{
    private _viewer : Viewer;

    constructor(viewer : Viewer){
        this._viewer = viewer;
        //.... AnimationCurves ....
        // Default animation curve is easeOut, without any bind it would be fc0
        this._viewer.bindAnimationCurve(AnimableValue.COULEUR, AnimationFuction.easeOut);
        this._viewer.bindAnimationCurve(AnimableValue.POSITION, AnimationFuction.easeOut);

        this.initAnimationItem()
    }

    private initAnimationItem(){
        let animationItem = document.getElementById("animationFunctionsGrid") as HTMLDivElement;
        let select = document.getElementById("animableSelect") as HTMLSelectElement
        let animationKeysValue = Object.keys(AnimableValue)
        for(let i=0; i<animationKeysValue.length/2; i++){
            let option = document.createElement("option")
            option.value = animationKeysValue.at(i).toString();
            option.innerText = animationKeysValue.at(i+animationKeysValue.length/2);
            option.setAttribute("animationFunction","easeOut");
            select.appendChild(option)
        }

        let optionAll = document.createElement("option");
        optionAll.value = "-1";
        optionAll.innerText = "TOUS";
        optionAll.setAttribute("animationFunction","easeOut");
        select.appendChild(optionAll);

        //Iterate over all the function in AnimationFunction
        for(let animFunction of Object.values(AnimationFuction)){
            let canvas = document.createElement("canvas") as HTMLCanvasElement;
            canvas.width = 80;
            canvas.height = 120;
            canvas.title = animFunction.name;

            let ctx = canvas.getContext("2d");
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#0a3b49";
            ctx.beginPath();
            let path = new Path2D()
            let offset = 0;
            let y = animFunction(1/canvas.width)*canvas.width
            let y_next = y
            for(let x = 1; x<canvas.width-2; x++){
                offset = y < offset ? y : offset
                path.moveTo(x,y);
                y_next = animFunction((x+1)/canvas.width)*canvas.width 
                path.lineTo(x+1 , y_next);
                y = y_next
            }
            ctx.setTransform(1,0,0,1, 0,-offset + 3);
            ctx.stroke(path);
            let container = document.createElement("div");
            container.id = animFunction.name;
            container.classList.add("afGridItem");
            if(animFunction.name == "easeOut")
                container.classList.add("active")
            container.appendChild(canvas);

            let name = document.createElement("h5");
            name.innerText = animFunction.name;
            container.appendChild(name)

            container.addEventListener("click", () => {
                let animableProperty = Number.parseInt(select.value);
                if(animableProperty == -1){
                    for(let i=0; i<animationKeysValue.length/2;i++){
                        this._viewer.bindAnimationCurve(i, animFunction);
                    }
                }else{
                    this._viewer.bindAnimationCurve(animableProperty, animFunction);
                }
                let predActive = document.getElementsByClassName("afGridItem active")[0]
                if(predActive){
                    predActive.classList.remove("active")
                }
                container.classList.add("active")
                select.children[select.selectedIndex].setAttribute("animationFunction",animFunction.name)
            });
            animationItem.appendChild(container);
        }
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
