import { WorkerTimers } from "./workerTimers.js";
export class SocketManager {
    // Singleton
    static _instance;
    timers;
    _requestDataMessage = "RequestData";
    _resetSimulationMessage = "ResetSimulation";
    _resetRandomSimulationMessage = "ResetRandomSimulation";
    _changeSimuMessage = "ChangeSimulation";
    _updateInitParams = "UpdateInitParams";
    _updateRulesMessage = "UpdateRule";
    _applyInteractionMessage = "ApplyInteraction";
    // These functions must be defined by the owner
    _onDataReceived;
    _onStart;
    _onStop;
    //..............................................
    _isConnected;
    _socket;
    _awaitingRequests;
    constructor() {
        this._onDataReceived = function (data) {
            console.log("Data received");
        };
        this._onStart = function () { };
        this._onStop = function () { };
        this._isConnected = false;
        this._awaitingRequests = [];
        this.timers = WorkerTimers.getInstance();
    }
    static getInstance() {
        if (!SocketManager._instance)
            SocketManager._instance = new SocketManager();
        return SocketManager._instance;
    }
    get isConnected() {
        return this._isConnected;
    }
    // setters
    set onDataReceived(fct) {
        this._onDataReceived = fct;
    }
    set onStart(fct) {
        this._onStart = fct;
    }
    set onStop(fct) {
        this._onStop = fct;
    }
    // private methods
    connectSocketEvents() {
        let self = this;
        this._socket.onmessage = function (e) {
            self.onMessage(e);
        };
        this._socket.onopen = function (e) {
            self.onOpen(e);
        };
        this._socket.onerror = function (e) {
            self.onError(e);
        };
        this._socket.onclose = function (e) {
            self.onClose(e);
        };
    }
    onMessage(e) {
        this.timers.stopReceivingTimer();
        var promise = e.data.text();
        promise.then(value => {
            this.timers.startParsingTimer();
            const data = JSON.parse(value);
            this.timers.stopParsingTimer();
            this._onDataReceived(data);
        });
    }
    onClose(e) {
        console.log("onClose");
        this._isConnected = false;
        console.debug('Socket closed unexpectedly', e);
    }
    onError(e) {
        console.log("onError");
        this._isConnected = false;
        console.error('Socket closed unexpectedly', e);
    }
    onOpen(e) {
        this._isConnected = true;
        console.debug('Socket opened');
        this._awaitingRequests.forEach(fct => { fct(); });
    }
    // public methods
    async connectSocket(url) {
        this._socket = new WebSocket(url);
        this.connectSocketEvents();
    }
    requestData() {
        this._socket.send(JSON.stringify({
            'message': this._requestDataMessage
        }));
        this.timers.startReceivingTimer();
    }
    resetSimulation() {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.resetSimulation.bind(this));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._resetSimulationMessage,
        }));
    }
    resetRandomSimulation() {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.resetRandomSimulation.bind(this));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._resetRandomSimulationMessage,
        }));
    }
    updateSimuRules(params) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.updateSimuRules.bind(this, params));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._updateRulesMessage,
            'params': params
        }));
    }
    updateInitParams(params) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.updateInitParams.bind(this, params));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._updateInitParams,
            'params': params
        }));
    }
    changeSimu(name) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.changeSimu.bind(this, name));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._changeSimuMessage,
            'simuName': name
        }));
    }
    apply_interaction(mask, interaction, id) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.apply_interaction.bind(this, mask, interaction, id));
            return;
        }
        let string = JSON.stringify({
            'message': this._applyInteractionMessage,
            'mask': Array.from(mask),
            'id': id,
            'interaction': interaction
        });
        this._socket.send(string);
    }
}
