import { TransformableValues } from "../transformableValues.js";
import { SocketManager } from "./socketManager.js";
import { StatesBuffer } from "./statesBuffer.js";
import { WorkerMessage, sendMessageToWindow, getMessageBody, getMessageHeader } from "./workerInterface.js";
class TransmissionWorker {
    _socketManager;
    _statesBuffer;
    constructor() {
        this._socketManager = SocketManager.getInstance();
        this._statesBuffer = new StatesBuffer();
        onmessage = this.onMessage.bind(this);
    }
    onMessage(e) {
        switch (getMessageHeader(e)) {
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
    async initSocket(url) {
        if (this._socketManager.isConnected)
            return;
        await this._socketManager.connectSocket(url);
    }
    async sendValues(waitAnotherStates = false) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        let isReshaped = this._statesBuffer.isReshaped;
        console.log("value is new :", this._statesBuffer.hasNewValue);
        let values = this._statesBuffer.values;
        console.log("value = ", values);
        if (waitAnotherStates) {
            while (!this._statesBuffer.hasNewValue) {
                await new Promise(resolve => setTimeout(resolve, 1));
            }
        }
        if (isReshaped) {
            sendMessageToWindow(WorkerMessage.VALUES_RESHAPED, values.toArray(), values.toArrayBuffers());
        }
        else {
            sendMessageToWindow(WorkerMessage.VALUES, values.toArray(), values.toArrayBuffers());
        }
    }
    async resetSimulation() {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetSimulation();
        while (!this._statesBuffer.hasNewValue) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        this.sendValues();
    }
    async waitSocketConnection() {
        while (!this._socketManager.isConnected) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
    }
    async updateSimulationRules(params) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.updateSimuRules(params);
    }
    async updateInitParams(params) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.updateInitParams(params);
        console.log("init param = ", params);
    }
    async applyInteraction(data) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._statesBuffer.flush();
        let values = TransformableValues.fromValuesAsArray(data.slice(1));
        this._socketManager.applyInteraction(data[0], values.getBackendValues());
        while (!this._statesBuffer.hasNewValue) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
        await this.sendValues(true);
    }
    async changeSimulation(nameSimu) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.changeSimu(nameSimu);
    }
}
const url = 'ws://'
    + self.location.host
    + '/ws/viewer/';
let worker = new TransmissionWorker();
worker.initSocket(url);
