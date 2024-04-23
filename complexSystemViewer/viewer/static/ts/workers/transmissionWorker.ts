import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
import { StatesBuffer } from "./statesBuffer.js";
import { WorkerTimers } from "./workerTimers.js";
import { WorkerMessage, sendMessageToWindow, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workerInterface.js";

class TransmissionWorker{
    private _socketManager : SocketManager;
    private _statesBuffer : StatesBuffer;

    public timers : WorkerTimers

    constructor(){
        this._socketManager = SocketManager.getInstance();
        this._statesBuffer = new StatesBuffer();
        onmessage = this.onMessage.bind(this);
        this.timers = WorkerTimers.getInstance();
    }

    private onMessage(e : MessageEvent<any>) : void {
        switch(getMessageHeader(e)){
            case WorkerMessage.INIT_SOCKET:
                this.initSocket(getMessageBody(e));
                break;
            case WorkerMessage.GET_VALUES:
                this.sendValues();
                break;
            case WorkerMessage.RESET:
                this.resetSimulation();
                break;
            case WorkerMessage.RESET_RANDOM:
                this.resetRandomSimulation();
                break;
            case WorkerMessage.UPDATE_RULES:
                this.updateSimulationRules(getMessageBody(e));
                break;
            case WorkerMessage.APPLY_INTERACTION:
                this.apply_interaction(getMessageBody(e)[2], getMessageBody(e)[0], getMessageBody(e)[1]);
                break;
            case WorkerMessage.CHANGE_SIMULATION:
                this.changeSimulation(getMessageBody(e));
                break;
            case WorkerMessage.UPDATE_INIT_PARAM:
                this.updateInitParams(getMessageBody(e));
        }
    }

    public async initSocket(url : string | URL){
        if (this._socketManager.isConnected)
            return;
        await this._socketManager.connectSocket(url);
        await this.waitSocketConnection();
        this._statesBuffer.requestState();
    }

    private async sendValues(waitAnotherStates : boolean = false){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        let isReshaped = this._statesBuffer.isReshaped;
        let values = this._statesBuffer.values;
        if (waitAnotherStates){
            await this.waitNewValues();
        }

        if (isReshaped){
            sendMessageToWindow(WorkerMessage.VALUES_RESHAPED, values.toArray(), values.toArrayBuffers());
        }
        else{            
            sendMessageToWindow(WorkerMessage.VALUES, values.toArray(), values.toArrayBuffers());
        }

        sendMessageToWindow(WorkerMessage.SET_TIMER, ["transformation", this.timers.transformationTimer]);
        sendMessageToWindow(WorkerMessage.SET_TIMER, ["parsing", this.timers.parsingTimer]);
        sendMessageToWindow(WorkerMessage.SET_TIMER, ["receiving", this.timers.receivingTimer]);

    }
    
    private async resetSimulation(){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetSimulation();
        await this.waitNewValues();
        this.sendValues();
    }

    private async resetRandomSimulation(){
        if (!this._socketManager.isConnected)
                await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetRandomSimulation();
        await this.waitNewValues();
        this.sendValues();
    }

    private async waitSocketConnection(){
        while (!this._socketManager.isConnected){
            await new Promise(resolve => setTimeout(resolve, 1));
        };
    }

    private async updateSimulationRules(params : any){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.updateSimuRules(params);
    }

    private async updateInitParams(params : any){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.updateInitParams(params);
    }

    private async apply_interaction(mask : Float32Array, interaction : string, id : number){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        
        this._statesBuffer.flush();
        this._socketManager.apply_interaction(mask, interaction, id);
        
        await this.waitNewValues();
        
        await this.sendValues(true);

    }

    private async changeSimulation(nameSimu : any){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        
        this._statesBuffer.flush();
        this._socketManager.changeSimu(nameSimu);
        
        await this.waitNewValues();
        
        this.sendValues();
    }

    private async waitNewValues(){
        while (!this._statesBuffer.hasNewValue){
            await new Promise(resolve => setTimeout(resolve, 1));
        };
    }
}

const url = 
'ws://'
+ self.location.host
+ '/ws/viewer/';

let worker = new TransmissionWorker();
worker.initSocket(url);
