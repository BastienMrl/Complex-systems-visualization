export class SocketHandler {
    // Singleton
    static _instance;
    _startMesssage = "Start";
    _stopMessage = "Stop";
    _requestDataMessage = "RequestData";
    // These functions must be defined by the owner
    _onDataReceived;
    _onStart;
    _onStop;
    //..............................................
    _isRunning;
    _socket;
    constructor() {
        this._onDataReceived = function (data) {
            console.log("Data received");
        };
        this._onStart = function () { };
        this._onStop = function () { };
    }
    static getInstance() {
        if (!SocketHandler._instance)
            SocketHandler._instance = new SocketHandler();
        return SocketHandler._instance;
    }
    // getter
    get isRunning() {
        return this._isRunning;
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
        this._socket.onopen = this.onOpen;
        this._socket.onerror = this.onError;
        this._socket.onclose = this.onClose;
    }
    onMessage(e) {
        const data = JSON.parse(e.data);
        this._onDataReceived(data);
    }
    onClose(e) {
        console.debug('Socket closed unexpectedly', e);
    }
    onError(e) {
        console.error('Socket closed unexpectedly', e);
    }
    onOpen(e) {
        console.debug('Socket opened');
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
}
