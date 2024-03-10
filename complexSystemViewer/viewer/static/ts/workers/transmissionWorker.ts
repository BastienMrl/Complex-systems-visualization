import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
import { StatesBuffer } from "./statesBuffer.js";
import { WorkerMessage, sendMessageToWindow, getMessageBody, getMessageHeader, sendMessageToWorker } from "./workerInterface.js";

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
                this.resetSimulation();
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
            case WorkerMessage.UPDATE_INIT_PARAM:
                this.updateInitParams(getMessageBody(e));
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
        let isReshaped = this._statesBuffer.isReshaped;
        let values = this._statesBuffer.values;

        if (isReshaped){
            sendMessageToWindow(WorkerMessage.VALUES_RESHAPED, values.toArray(), values.toArrayBuffers());
        }
        else{            
            sendMessageToWindow(WorkerMessage.VALUES, values.toArray(), values.toArrayBuffers());
        }
    }
    
    private async resetSimulation(){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetSimulation();
        while(!this._statesBuffer.hasNewValue){
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
        this._socketManager.updateSimuRules(params);
    }

    private async updateInitParams(params : any){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.updateInitParams(params);
        console.log("init param = ", params)
    }

    private async applyInteraction(data : Array<Float32Array>){
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        
        this._statesBuffer.flush();
        
        let values = TransformableValues.fromValuesAsArray(data.slice(1));
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
