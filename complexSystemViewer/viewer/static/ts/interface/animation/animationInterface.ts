import { ViewerManager } from "../../viewerManager.js";
import { AnimationFunction } from "./animationFunctions.js";
import { AnimableValue } from "../../shaderUtils.js";

export class AnimationInterface{
    private _viewer : ViewerManager;

    private MAX_REFRESH_RATE = 30.;
    private MIN_REFRESH_RATE = 0.5;
    private REFRESH_STEP = 0.5;
    private DEFAULT_REFRESH_RATE = 6.;

    constructor(viewer : ViewerManager){
        this._viewer = viewer;
        //.... AnimationCurves ....
        // Default animation curve is easeOut, without any bind it would be fc0
        this._viewer.bindAnimationCurve(AnimableValue.COLOR, AnimationFunction.linear);
        this._viewer.bindAnimationCurve(AnimableValue.POSITION, AnimationFunction.linear);

        this.initAnimationItem()
    }

    private initAnimationItem(){
        let animationItem = document.getElementById("animationFunctionsGrid") as HTMLDivElement;
        let select = document.getElementById("animableSelect") as HTMLSelectElement
        let animationKeysValue = Object.values(AnimableValue);
        for(let i=0; i<animationKeysValue.length/2; i++){
            let option = document.createElement("option")
            option.value = animationKeysValue[i].toString();
            option.innerText = animationKeysValue[i].toString();
            option.setAttribute("animationFunction","easeOut");
            select.appendChild(option)
        }

        let optionAll = document.createElement("option");
        optionAll.value = "-1";
        optionAll.innerText = "ALL";
        optionAll.setAttribute("animationFunction","easeOut");
        select.appendChild(optionAll);

        //Iterate over all the function in AnimationFunction
        for(let animFunction of Object.values(AnimationFunction)){
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
                let animableProperty = this.getAnimableValueFromString(select.value);
                if(animableProperty == undefined){
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

    private getAnimableValueFromString(name : string) : number {
        switch (name) {
            case "COLOR" : return AnimableValue.COLOR;
            case "POSITION" : return AnimableValue.POSITION;
            case "ROTATION" : return AnimableValue.ROTATION;
            case "SCALING" : return AnimableValue.SCALING;
        }
    }

    public setDurationElement(element : HTMLElement){
        let input = document.getElementById("inputTimer") as HTMLInputElement;
        let label = document.getElementById("labelTimer") as HTMLLabelElement;
        input.min =   `${this.MIN_REFRESH_RATE}`;
        input.max =   `${this.MAX_REFRESH_RATE}`;
        input.step =  `${this.REFRESH_STEP}`;
        input.value = `${this.DEFAULT_REFRESH_RATE}`;
        label.innerHTML = `<strong>${input.value}</strong> steps per second`;
        this._viewer.animationDuration = (1. / Number(input.value));
        input.addEventListener("input", () => {
            label.innerHTML = `<strong>${input.value}</strong> steps per second`;
            this._viewer.animationDuration = (1. / Number(input.value));
        });
        element.style.display = 'none';
    }

}