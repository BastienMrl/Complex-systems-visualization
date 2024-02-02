class SocketHandler {
    // Singleton
    static #instance;

    #startMesssage = "Start";
    #stopMessage = "Stop";

    // These functions must be defined by the owner
    onDataReceived;
    onStart;
    onStop;
    //..............................................

    #isRunning;
    #socket;


    constructor() {
        if (SocketHandler.#instance) {
            return SocketHandler.#instance
        }
        SocketHandler.#instance = this;


        this.onDataReceived = function(data) {
            console.log("Data received");
        }
        this.onStart = function() {};
        this.onStop = function() {};
        return this
    }
    
    async connectSocket(url){
        this.#socket = new WebSocket(url);
        this.#isRunning = false;
        this.#connectSocketEvents()
    }

    isRunning(){
        return this.#isRunning
    }

    #connectSocketEvents(){
        let self = this;
        this.#socket.onmessage = function(e){
            self.#onMessage(e);
        }
        this.#socket.onopen = this.#onOpen;
        this.#socket.onerror = this.#onError;
        this.#socket.onclose = this.#onClose;
    }

    // params could be "Model" + "instance parameters"
    start(params){
        if (this.#isRunning) return;
        this.#socket.send(JSON.stringify({
            'message': this.#startMesssage,
            'params' : params
        }));
        this.#isRunning = true;
        this.onStart();
    }

    stop(){
        if (!this.#isRunning) return;
        this.#socket.send(JSON.stringify({
            'message': this.#stopMessage
        }));
        this.#isRunning = false;
        this.onStop();
    }

    #onMessage(e) {
        const data = JSON.parse(e.data);
        this.onDataReceived(data);
    }

    #onClose(e) {
        console.debug('Socket closed unexpectedly', e);
    }

    #onError(e){
        console.error('Socket closed unexpectedly', e)
    }

    #onOpen(e) {
        console.debug('Socket opened');
    }
}

export { SocketHandler }