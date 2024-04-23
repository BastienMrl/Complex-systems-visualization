import { SocketManager } from "./socketManager.js";
import { StatesBuffer } from "./statesBuffer.js";
import { WorkerTimers } from "./workerTimers.js";
import { WorkerMessage, sendMessageToWindow, getMessageBody, getMessageHeader } from "./workerInterface.js";
class TransmissionWorker {
    _socketManager;
    _statesBuffer;
    timers;
    constructor() {
        this._socketManager = SocketManager.getInstance();
        this._statesBuffer = new StatesBuffer();
        onmessage = this.onMessage.bind(this);
        this.timers = WorkerTimers.getInstance();
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
    async initSocket(url) {
        if (this._socketManager.isConnected)
            return;
        await this._socketManager.connectSocket(url);
        await this.waitSocketConnection();
        this._statesBuffer.requestState();
    }
    async sendValues(waitAnotherStates = false) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        let isReshaped = this._statesBuffer.isReshaped;
        let values = this._statesBuffer.values;
        if (waitAnotherStates) {
            await this.waitNewValues();
        }
        if (isReshaped) {
            sendMessageToWindow(WorkerMessage.VALUES_RESHAPED, values.toArray(), values.toArrayBuffers());
        }
        else {
            sendMessageToWindow(WorkerMessage.VALUES, values.toArray(), values.toArrayBuffers());
        }
        sendMessageToWindow(WorkerMessage.SET_TIMER, ["transformation", this.timers.transformationTimer]);
        sendMessageToWindow(WorkerMessage.SET_TIMER, ["parsing", this.timers.parsingTimer]);
        sendMessageToWindow(WorkerMessage.SET_TIMER, ["receiving", this.timers.receivingTimer]);
    }
    async resetSimulation() {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetSimulation();
        await this.waitNewValues();
        this.sendValues();
    }
    async resetRandomSimulation() {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        sendMessageToWindow(WorkerMessage.RESET);
        this._statesBuffer.flush();
        this._socketManager.resetRandomSimulation();
        await this.waitNewValues();
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
    }
    async apply_interaction(mask, interaction, id) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._statesBuffer.flush();
        this._socketManager.apply_interaction(mask, interaction, id);
        await this.waitNewValues();
        await this.sendValues(true);
    }
    async changeSimulation(nameSimu) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._statesBuffer.flush();
        this._socketManager.changeSimu(nameSimu);
        await this.waitNewValues();
        this.sendValues();
    }
    async waitNewValues() {
        while (!this._statesBuffer.hasNewValue) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }
        ;
    }
}
const url = 'ws://'
    + self.location.host
    + '/ws/viewer/';
let worker = new TransmissionWorker();
worker.initSocket(url);
