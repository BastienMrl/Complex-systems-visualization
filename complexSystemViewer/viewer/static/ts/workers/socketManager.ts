


export class SocketManager {
    // Singleton
    private static _instance : SocketManager;

    private _requestDataMessage : string = "RequestData";
    private _resetSimulationMessage : string = "ResetSimulation";
    private _changeSimuMessage : string = "ChangeSimulation";
    private _updateInitParams : string = "UpdateInitParams";
    private _updateRulesMessage : string = "UpdateRule";
    private _applyInteractionMessage : string = "ApplyInteraction";
    

    // These functions must be defined by the owner
    private _onDataReceived : (data : any) => void;
    private _onStart : () => void;
    private _onStop : () => void;
    //..............................................

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
        this.connectSocketEvents();
    }

    public requestData(){
        this._socket.send(JSON.stringify({
            'message' : this._requestDataMessage
        }));
    }

    public resetSimulation(){
        if (!this._isConnected){
            this._awaitingRequests.push(this.resetSimulation.bind(this));
            return;
        };
        
        this._socket.send(JSON.stringify({
            'message' : this._resetSimulationMessage,
        }));
    }

    public updateSimuRules(params: any){
        if (!this._isConnected){
            this._awaitingRequests.push(this.updateSimuRules.bind(this, params));
            return;
        };
        this._socket.send(JSON.stringify({
            'message' : this._updateRulesMessage,
            'params' : params
        }));
    }

    public updateInitParams(params: any){
        if (!this._isConnected){
            this._awaitingRequests.push(this.updateInitParams.bind(this, params));
            return;
        };
        this._socket.send(JSON.stringify({
            'message' : this._updateInitParams,
            'params' : params
        }));
    }

    public changeSimu(name: string){
        if (!this._isConnected){
            this._awaitingRequests.push(this.changeSimu.bind(this, name));
            return;
        };
        this._socket.send(JSON.stringify({
            'message' : this._changeSimuMessage,
            'simuName' : name
        }));
    }

    public applyInteraction(mask : Float32Array, currentValues : Array<Float32Array>, id : string){
        if (!this._isConnected){
            this._awaitingRequests.push(this.applyInteraction.bind(this, mask));
            return;
        }
        let values = new Array(currentValues.length);
        currentValues.forEach((e, i) => {
            values[i] = Array.from(e);
        });

        let string = JSON.stringify({
            'message' : this._applyInteractionMessage,
            'mask' : Array.from(mask),
            'currentStates' : values,
            'interaction' : id
        });

        
        this._socket.send(string);
    }
    
}