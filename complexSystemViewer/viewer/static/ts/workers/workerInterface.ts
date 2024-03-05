export enum WorkerMessage {
    INIT_SOCKET = "init",
    GET_VALUES = "get",
    RESET = "reset",
    VALUES = "values",
    READY = "ready",
    UPDATE_RULES = "update_r",
    APPLY_INTERACTION = "send_interaction"
}

export function sendMessageToWorker(worker : Worker, header : WorkerMessage, message? : any, transfer? : Transferable[]){
    if (transfer != undefined)
        worker.postMessage([header, message], {transfer : transfer});
    else{
        worker.postMessage([header, message]);
    }
}

export function sendMessageToWindow(header : WorkerMessage, data? : any, transfer? : Transferable[]){
    if (transfer != undefined)
        postMessage([header, data], {transfer : transfer})
    else 
        postMessage([header, data]);
}

export function getMessageHeader(e : MessageEvent<any>) : WorkerMessage{
    return e.data[0] as WorkerMessage;
}

export function getMessageBody(e : MessageEvent<any>) : any{
    return e.data[1];
}