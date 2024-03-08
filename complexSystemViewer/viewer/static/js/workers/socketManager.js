export class SocketManager {
    // Singleton
    static _instance;
    _requestDataMessage = "RequestData";
    _resetSimulationMessage = "ResetSimulation";
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
        var promise = e.data.text();
        promise.then(value => {
            const data = JSON.parse(value);
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
    }
    resetSimulation(params) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.resetSimulation.bind(this, params));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._resetSimulationMessage,
            'params': params
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
    applyInteraction(mask, currentValues) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.applyInteraction.bind(this, mask));
            return;
        }
        let values = new Array(currentValues.length);
        currentValues.forEach((e, i) => {
            values[i] = Array.from(e);
        });
        let string = JSON.stringify({
            'message': this._applyInteractionMessage,
            'mask': Array.from(mask),
            'currentStates': values
        });
        this._socket.send(string);
    }
}
