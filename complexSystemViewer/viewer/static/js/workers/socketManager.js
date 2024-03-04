export class SocketManager {
    // Singleton
    static _instance;
    _startMesssage = "Start";
    _stopMessage = "Stop";
    _requestDataMessage = "RequestData";
    _requestEmptyGridMessage = "EmptyGrid";
    _changeRulesMessage = "ChangeRules";
    _applyInteractionMessage = "ApplyInteraction";
    // These functions must be defined by the owner
    _onDataReceived;
    _onStart;
    _onStop;
    //..............................................
    _isRunning;
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
    // getter
    get isRunning() {
        return this._isRunning;
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
        this._isRunning = false;
        this.connectSocketEvents();
    }
    // params could be "Model" + "instance parameters"
    start(params) {
        if (this._isRunning)
            return;
        this._socket.send(JSON.stringify({
            'message': this._startMesssage,
            'params': params
        }));
        this._isRunning = true;
        this._onStart();
    }
    stop() {
        if (!this._isRunning)
            return;
        this._socket.send(JSON.stringify({
            'message': this._stopMessage
        }));
        this._isRunning = false;
        this._onStop();
    }
    requestData() {
        if (!this._isRunning)
            return;
        this._socket.send(JSON.stringify({
            'message': this._requestDataMessage
        }));
    }
    requestEmptyInstance(params) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.requestEmptyInstance.bind(this, params));
            return;
        }
        ;
        if (this._isRunning)
            return;
        this._socket.send(JSON.stringify({
            'message': this._requestEmptyGridMessage,
            'params': params
        }));
    }
    changeSimuRules(params) {
        console.log(this);
        if (!this._isConnected) {
            this._awaitingRequests.push(this.changeSimuRules.bind(this, params));
            return;
        }
        ;
        this._socket.send(JSON.stringify({
            'message': this._changeRulesMessage,
            'params': params
        }));
    }
    applyInteraction(mask) {
        if (!this._isConnected) {
            this._awaitingRequests.push(this.applyInteraction.bind(this, mask));
            return;
        }
        this._socket.send(JSON.stringify({
            'message': this._applyInteractionMessage,
            'mask': Array.from(mask)
        }));
    }
}
