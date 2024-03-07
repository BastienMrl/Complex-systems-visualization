export var WorkerMessage;
(function (WorkerMessage) {
    WorkerMessage["INIT_SOCKET"] = "init";
    WorkerMessage["GET_VALUES"] = "get";
    WorkerMessage["RESET"] = "reset";
    WorkerMessage["VALUES"] = "values";
    WorkerMessage["READY"] = "ready";
    WorkerMessage["UPDATE_RULES"] = "update_r";
    WorkerMessage["APPLY_INTERACTION"] = "send_interaction";
    WorkerMessage["CHANGE_SIMULATION"] = "change_simulation";
})(WorkerMessage || (WorkerMessage = {}));
export function sendMessageToWorker(worker, header, message, transfer) {
    if (transfer != undefined)
        worker.postMessage([header, message], { transfer: transfer });
    else {
        worker.postMessage([header, message]);
    }
}
export function sendMessageToWindow(header, data, transfer) {
    if (transfer != undefined)
        postMessage([header, data], { transfer: transfer });
    else
        postMessage([header, data]);
}
export function getMessageHeader(e) {
    return e.data[0];
}
export function getMessageBody(e) {
    return e.data[1];
}
