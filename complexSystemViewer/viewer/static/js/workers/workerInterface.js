export var WorkerMessage;
(function (WorkerMessage) {
    WorkerMessage["INIT_SOCKET"] = "init";
    WorkerMessage["GET_VALUES"] = "get";
    WorkerMessage["RESET"] = "reset";
    WorkerMessage["VALUES"] = "values";
    WorkerMessage["READY"] = "ready";
    WorkerMessage["UPDATE_RULES"] = "update_r";
})(WorkerMessage || (WorkerMessage = {}));
export function sendMessageToWorker(worker, header, message) {
    let data = [header, message];
    worker.postMessage(data);
}
export function sendMessageToWindow(header, data, transfer) {
    if (transfer != undefined)
        postMessage([header, data], { transfer: transfer });
    else
        postMessage([header, data]);
}
export function getMessageHeader(data) {
    return data[0];
}
export function getMessageBody(data) {
    return data[1];
}
