export enum WorkerMessage {
    INIT_SOCKET = "init",
    GET_VALUES = "get",
    RESET = "reset",
    VALUES = "values",
    READY = "ready",
    UPDATE_RULES = "update_r"
}

export function sendMessageToWorker(worker : Worker, header : WorkerMessage, message? : any){
    let data = [header, message];
    worker.postMessage(data);
}

export function sendMessageToWindow(header : WorkerMessage, data? : any, transfer? : Transferable[]){
    if (transfer != undefined)
        postMessage([header, data], {transfer : transfer})
    else 
        postMessage([header, data]);
}

export function getMessageHeader(data : any) : WorkerMessage{
    return data[0] as WorkerMessage;
}

export function getMessageBody(data : any) : any{
    return data[1];
}