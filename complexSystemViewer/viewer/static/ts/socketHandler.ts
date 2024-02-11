export class SocketHandler {
    // Singleton
    private static _instance : SocketHandler;

    private _startMesssage : string = "Start";
    private _stopMessage : string = "Stop";
    private _requestDataMessage : string = "RequestData";

    // These functions must be defined by the owner
    private _onDataReceived : (data : any) => void;
    private _onStart : () => void;
    private _onStop : () => void;
    //..............................................

    private _isRunning : boolean;
    private _socket : WebSocket;


    private constructor() {
        this._onDataReceived = function(data) {
            console.log("Data received");
        }
        this._onStart = function() {};
        this._onStop = function() {};
    }
    
    public static getInstance() : SocketHandler {
        if (!SocketHandler._instance)
            SocketHandler._instance = new SocketHandler();
        return SocketHandler._instance;
    }
    
    // getter
    public get isRunning() : boolean{
        return this._isRunning;
    
    }

    // setters
    public set onDataReceived(fct : (data : any) => void){
        this._onDataReceived = fct;
    }

    public set onStart(fct : () => void){
        this._onStart = fct;
    }

    public set onStop(fct : () => void){
        this._onStop = fct;
    }
    
    // private methods
    private connectSocketEvents(){
        let self : this = this;
        this._socket.onmessage = function(e : MessageEvent<any>){
            self.onMessage(e);
        }
        this._socket.onopen = this.onOpen;
        this._socket.onerror = this.onError;
        this._socket.onclose = this.onClose;
    }

    private onMessage(e : MessageEvent<any>) {
        const data : any = JSON.parse(e.data);
        this._onDataReceived(data);
    }

    private onClose(e : CloseEvent) {
        console.debug('Socket closed unexpectedly', e);
    }

    private onError(e : Event){
        console.error('Socket closed unexpectedly', e)
    }

    private onOpen(e : Event) {
        console.debug('Socket opened');
    }


    // public methods
    public async connectSocket(url : string | URL){
        this._socket = new WebSocket(url);
        this._isRunning = false;
        this.connectSocketEvents();
    }

    // params could be "Model" + "instance parameters"
    public start(params){
        if (this._isRunning) return;
        this._socket.send(JSON.stringify({
            'message': this._startMesssage,
            'params' : params
        }));
        this._isRunning = true;
        this._onStart();
    }
    
    public stop(){
        if (!this._isRunning) return;
        this._socket.send(JSON.stringify({
            'message': this._stopMessage
        }));
        this._isRunning = false;
        this._onStop();
    }

    public requestData(){
        if (!this._isRunning) return;
        this._socket.send(JSON.stringify({
            'message' : this._requestDataMessage
        }));
    }
    
}