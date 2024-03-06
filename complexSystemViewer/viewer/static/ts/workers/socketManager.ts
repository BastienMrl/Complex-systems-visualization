


export class SocketManager {
    // Singleton
    private static _instance : SocketManager;

    private _startMesssage : string = "Start";
    private _stopMessage : string = "Stop";
    private _requestDataMessage : string = "RequestData";
    private _requestEmptyGridMessage : string = "EmptyGrid";
    private _changeRulesMessage : string = "ChangeRules";
    private _applyInteractionMessage : string = "ApplyInteraction";

    // These functions must be defined by the owner
    private _onDataReceived : (data : any) => void;
    private _onStart : () => void;
    private _onStop : () => void;
    //..............................................

    private _isRunning : boolean;
    private _isConnected : boolean;
    private _socket : WebSocket;

    private _awaitingRequests : ((param? : any) => void)[];


    private constructor() {
        this._onDataReceived = function(data) {
            console.log("Data received");
        }
        this._onStart = function() {};
        this._onStop = function() {};
        this._isConnected = false;
        this._awaitingRequests = [];
    }
    
    public static getInstance() : SocketManager {
        if (!SocketManager._instance)
            SocketManager._instance = new SocketManager();
        return SocketManager._instance;
    }
    
    // getter
    public get isRunning() : boolean{
        return this._isRunning;
    
    }

    public get isConnected() : boolean{
        return this._isConnected;
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
        };
        this._socket.onopen = function(e : Event){
            self.onOpen(e);
        };
        this._socket.onerror = function(e : Event){
            self.onError(e);
        };
        this._socket.onclose = function(e : CloseEvent){
            self.onClose(e);
        };
    }

    private onMessage(e : MessageEvent<any>) {
        var promise = e.data.text()
        promise.then(value => {
            const data : any = JSON.parse(value);
            this._onDataReceived(data);
        })
        
    }

    private onClose(e : CloseEvent) {
        console.log("onClose");
        this._isConnected = false;
        console.debug('Socket closed unexpectedly', e);
    }

    private onError(e : Event){
        console.log("onError");
        this._isConnected = false;
        console.error('Socket closed unexpectedly', e)
    }

    private onOpen(e : Event) {
        this._isConnected = true;
        console.debug('Socket opened');
        this._awaitingRequests.forEach(fct => { fct(); });
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

    public requestEmptyInstance(params : any){
        if (!this._isConnected){
            this._awaitingRequests.push(this.requestEmptyInstance.bind(this, params));
            return;
        };
        if (this._isRunning) return;
        this._socket.send(JSON.stringify({
            'message' : this._requestEmptyGridMessage,
            'params' : params
        }));
    }

    public changeSimuRules(params: any){
        console.log(this);
        if (!this._isConnected){
            this._awaitingRequests.push(this.changeSimuRules.bind(this, params));
            return;
        };
        this._socket.send(JSON.stringify({
            'message' : this._changeRulesMessage,
            'params' : params
        }));
    }

    public applyInteraction(mask : Float32Array, currentValues : Array<Float32Array>){
        if (!this._isConnected){
            this._awaitingRequests.push(this.applyInteraction.bind(this, mask));
            return;
        }
        this._socket.send(JSON.stringify({
            'message' : this._applyInteractionMessage,
            'mask' : Array.from(mask),
            'currentStates' : currentValues
        }));
    }
    
}