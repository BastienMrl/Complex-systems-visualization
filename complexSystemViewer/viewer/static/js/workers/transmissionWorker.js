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
        console.log(getMessageHeader(e));
        switch (getMessageHeader(e)) {
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
                this.changeSimulationRules(getMessageBody(e));
                break;
            case WorkerMessage.APPLY_INTERACTION:
                this.applyInteraction(getMessageBody(e));
                break;
        }
    }
    async initSocket(url) {
        if (this._socketManager.isConnected)
            return;
        await this._socketManager.connectSocket(url);
    }
    async sendValues() {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        let values = this._statesBuffer.values;
        sendMessageToWindow(WorkerMessage.VALUES, [values.states, values.translations], [values.states.buffer, values.translations.buffer]);
    }
    async resetSimulation(nbElements) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._statesBuffer.initializeElements(nbElements);
        while (!this._statesBuffer.isReady) {
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
    async changeSimulationRules(params) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.changeSimuRules(params);
    }
    async applyInteraction(mask) {
        if (!this._socketManager.isConnected)
            await this.waitSocketConnection();
        this._socketManager.applyInteraction(mask);
    }
}
const url = 'ws://'
    + self.location.host
    + '/ws/viewer/';
let worker = new TransmissionWorker();
worker.initSocket(url);
