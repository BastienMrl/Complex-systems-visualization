import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
import { StatesBuffer } from "./statesBuffer.js";
import { WorkerMessage, sendMessageToWindow, getMessageBody, getMessageHeader } from "./workerInterface.js";

class TransmissionWorker{
    private _socketManager : SocketManager;
    private _statesBuffer : StatesBuffer;

    constructor(){
        this._socketManager = SocketManager.getInstance();
        this._statesBuffer = new StatesBuffer();
        onmessage = this.onMessage.bind(this);
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
                this.resetSimulation(getMessageBody(e));
                break;
            case WorkerMessage.UPDATE_RULES:
                this.updateSimulationRules(getMessageBody(e));
                break;
            case WorkerMessage.APPLY_INTERACTION:
                this.applyInteraction(getMessageBody(e));
                break;
            case WorkerMessage.CHANGE_SIMULATION:
                this.changeSimulation(getMessageBody(e));
                break;
        }
    }

    public async initSocket(url : string | URL){
        if (this._socketManager.isConnected)
            return;
        await this._socketManager.connectSocket(url);
    }

    private async sendValues(){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        let values = this._statesBuffer.values;
        sendMessageToWindow(WorkerMessage.VALUES, values.toArray(), values.toArrayBuffers());
    }
    
    private async resetSimulation(nbElements : number){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._statesBuffer.initializeElements(nbElements);
        while(!this._statesBuffer.isReady){
            await new Promise(resolve => setTimeout(resolve, 1));
        }
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
        this._socketManager.updateSimuRules(params)
    }

    private async applyInteraction(data : Array<Float32Array>){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        
        this._statesBuffer.flush();
        
        let values = TransformableValues.fromArray(data.slice(1));
        this._socketManager.applyInteraction(data[0], values.getBackendValues());
        
        while (!this._statesBuffer.hasNewValue){
            await new Promise(resolve => setTimeout(resolve, 1));
        };

        await this.sendValues();
    }

    private async changeSimulation(nameSimu : any){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        
        this._socketManager.changeSimu(nameSimu);
    }
}

const url = 
'ws://'
+ self.location.host
+ '/ws/viewer/';

let worker = new TransmissionWorker();
worker.initSocket(url);
