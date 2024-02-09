import { Viewer } from "./viewer.js";
import { SocketHandler } from "./socketHandler.js";
import { UserEventHandler } from "./userEventHandler.js"

async function main(){
    let canvas : HTMLCanvasElement | null = document.getElementById("c") as HTMLCanvasElement | null;
    if (canvas == null) {
        throw "Could not find canvas";
    }
    canvas.height = canvas.clientHeight;
    canvas.width = canvas.clientWidth;
    
    
    let viewer = new Viewer("c");
    let userEventHandler = UserEventHandler.getInstance();
    userEventHandler.initHandlers(viewer);
    let nbInstances = 10 * 10;
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag", nbInstances);

    const url = 
        'ws://'
        + window.location.host
        + '/ws/viewer/';

    let socketHandler = SocketHandler.getInstance();

    // for instance, data is an array of bool
    socketHandler.onDataReceived = function(data) {
        viewer.updateState(data);
    }


    socketHandler.connectSocket(url);

    (document.querySelector('#buttonPlay') as HTMLButtonElement).onclick = function(e) {
        if (!socketHandler.isRunning){
            socketHandler.start(nbInstances);
            console.log("START");
        }
    };

    (document.querySelector('#buttonPause') as HTMLButtonElement).onclick = function(e) {
        if (socketHandler.isRunning){
            socketHandler.stop();
            console.log(socketHandler);
        }
    };

    
    function loop(time : number){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

window.onload = function () {
    main()
}