import { Viewer } from "../viewer.js";
import { TransformerBuilder } from "../transformer/transformerBuilder.js";
import { InputType } from "../transformer/inputType.js";
import { TransformType } from "../transformer/transformType.js";

export class TransformersInterface {
    private _viewer : Viewer;
    private _currentTransformerBuilder : TransformerBuilder;
    private _nbChannels : number;

    public constructor(viewer : Viewer){
        this._viewer = viewer;
        this._currentTransformerBuilder = new TransformerBuilder();
        this._nbChannels = 1;
        this._viewer.selectionManager.setTransformer(this._currentTransformerBuilder);
    }

    public setNumberOfStatesOutput(nb : number){
        let oldNumber = this._nbChannels;
        this._nbChannels = nb;
        document.querySelectorAll("div[transformer]").forEach((e) => {
            let select = e.getElementsByClassName("visualizationInput")[0].querySelector("select");
            if (oldNumber > this._nbChannels){
                for (let i = oldNumber - 1; i > this._nbChannels - 1; --i){
                    let selector = "option[value=" + this.getInputNameFromChannelIdx(i) + "]";
                    select.querySelector(selector).remove();
                }
            }
            else{
                for(let i = oldNumber; i < this._nbChannels; ++i){
                    let text = this.getInputNameFromChannelIdx(i);
                    let option = new Option(text, text);
                    select.add(option);
                }
            }

        })
    }

    private addInputTypeOptionElements(selectElement : HTMLSelectElement){
        let selected = this.getInputType(selectElement);
        let addOption = (text : string) => {
            let option = new Option(text, text);
            selectElement.add(option);
        }
        
        if (selected != InputType.POSITION_X)
            addOption("POSITION_X");
        if (selected != InputType.POSITION_Y)
            addOption("POSITION_Y");
        if (selected != InputType.POSITION_Z)
            addOption("POSITION_Z");
        for (let i = 0; i < this._nbChannels; i++){
            if (selectElement.value != this.getInputNameFromChannelIdx(i))
                addOption(this.getInputNameFromChannelIdx(i));
        }
    }

    public addTransformerFromElement(element : HTMLElement){
        const inputElement = this.getInputTypeElement(element);
        const inputType = this.getInputType(inputElement);
        this.addInputTypeOptionElements(inputElement);

        const deleteButton = (element.getElementsByClassName("deleteButton")[0] as HTMLButtonElement);
        
        const transformType = this.getTransformType(element);
        const paramsElements = this.getParamsElements(element);


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
            case "SCALING":
                return TransformType.SCALING;
            case "ROTATION_X":
                return TransformType.ROTATION_X;
            case "ROTATION_Y":
                return TransformType.ROTATION_Y;
            case "ROTATION_Z":
                return TransformType.ROTATION_Z;    
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
            case "STATE_1": 
                return InputType.STATE_1;
            case "STATE_2":
                return InputType.STATE_2;
            case "STATE_3":
                return InputType.STATE_3;
        }
    }

    public getInputNameFromChannelIdx(idx : number) : string{
        switch (idx){
            case 0:
                return "STATE_0";
            case 1:
                return "STATE_1";
            case 2:
                return "STATE_2";
            case 3:
                return "STATE_3";
            default:
                return "STATE_0";
        }
    }

    // TODO : fill with right ids
    private getParamsElements(parent : HTMLElement){
        switch (this.getTransformType(parent)){
            case TransformType.COLOR:
                let colorAliveInput = parent.querySelector("input[paramId=c1]") as HTMLInputElement;
                let colorDeadInput = parent.querySelector("input[paramId=c0]") as HTMLInputElement;     
                return [colorDeadInput, colorAliveInput];
            case TransformType.SCALING :
            case TransformType.COLOR_R :
            case TransformType.COLOR_G :
            case TransformType.COLOR_B :
            case TransformType.ROTATION_X:
            case TransformType.ROTATION_Y:
            case TransformType.ROTATION_Z:
                let min = parent.querySelector("input[paramId=range_min]") as HTMLInputElement;
                let max = parent.querySelector("input[paramId=range_max]") as HTMLInputElement;
                return [min, max];
            case TransformType.POSITION_X :
            case TransformType.POSITION_Y :
            case TransformType.POSITION_Z :
                return [parent.querySelector("input[paramId=factor]") as HTMLInputElement];
        }
    }
}