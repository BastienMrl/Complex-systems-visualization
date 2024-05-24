import { TransformerBuilder } from "../transformer/transformerBuilder.js";
import { InputType } from "../transformer/inputType.js";
import { TransformType } from "../transformer/transformType.js";
export class TransformersInterface {
    _viewer;
    _currentTransformerBuilder;
    _nbChannels;
    constructor(viewer) {
        this._viewer = viewer;
        this._currentTransformerBuilder = new TransformerBuilder();
        this._nbChannels = 1;
    }
    setNumberOfStatesOutput(nb) {
        let oldNumber = this._nbChannels;
        this._nbChannels = nb;
        document.querySelectorAll("div[transformer]").forEach((e) => {
            let select = e.getElementsByClassName("visualizationInput")[0].querySelector("select");
            if (oldNumber > this._nbChannels) {
                for (let i = oldNumber - 1; i > this._nbChannels - 1; --i) {
                    let selector = "option[value=" + this.getInputNameFromChannelIdx(i) + "]";
                    select.querySelector(selector).remove();
                }
            }
            else {
                for (let i = oldNumber; i < this._nbChannels; ++i) {
                    let text = this.getInputNameFromChannelIdx(i);
                    let option = new Option(text, text);
                    select.add(option);
                }
            }
        });
    }
    addInputTypeOptionElements(selectElement) {
        let selected = this.getInputType(selectElement);
        let addOption = (text) => {
            let option = new Option(text, text);
            selectElement.add(option);
        };
        if (selected != InputType.POSITION_X)
            addOption("POSITION_X");
        if (selected != InputType.POSITION_Y)
            addOption("POSITION_Y");
        if (selected != InputType.POSITION_Z)
            addOption("POSITION_Z");
        for (let i = 0; i < this._nbChannels; i++) {
            if (selectElement.value != this.getInputNameFromChannelIdx(i))
                addOption(this.getInputNameFromChannelIdx(i));
        }
    }
    addTransformerFromElement(element) {
        const inputElement = this.getInputTypeElement(element);
        const inputType = this.getInputType(inputElement);
        this.addInputTypeOptionElements(inputElement);
        const deleteButton = element.getElementsByClassName("deleteButton")[0];
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
        if (deleteButton) {
            deleteButton.addEventListener("click", () => {
                this._currentTransformerBuilder.removeTransformer(id);
                deleteButton.parentElement.remove();
                this.updateProgram();
            });
        }
    }
    updateProgram() {
        this._viewer.updateProgamsTransformers(this._currentTransformerBuilder);
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
            case "STATE_1":
                return InputType.STATE_1;
            case "STATE_2":
                return InputType.STATE_2;
            case "STATE_3":
                return InputType.STATE_3;
        }
    }
    getInputNameFromChannelIdx(idx) {
        switch (idx) {
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
    getParamsElements(parent) {
        switch (this.getTransformType(parent)) {
            case TransformType.COLOR:
                let colorAliveInput = parent.querySelector("input[paramId=c1]");
                let colorDeadInput = parent.querySelector("input[paramId=c0]");
                return [colorDeadInput, colorAliveInput];
            case TransformType.SCALING:
            case TransformType.COLOR_R:
            case TransformType.COLOR_G:
            case TransformType.COLOR_B:
            case TransformType.ROTATION_X:
            case TransformType.ROTATION_Y:
            case TransformType.ROTATION_Z:
                let min = parent.querySelector("input[paramId=range_min]");
                let max = parent.querySelector("input[paramId=range_max]");
                return [min, max];
            case TransformType.POSITION_X:
            case TransformType.POSITION_Y:
            case TransformType.POSITION_Z:
                return [parent.querySelector("input[paramId=factor]")];
        }
    }
}
