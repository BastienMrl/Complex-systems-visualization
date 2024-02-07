import { Viewer } from "./viewer.js"
import { SocketHandler } from "./socketHandler.js";
import { UserEventHandler } from "./userEventHandler.js"

async function main(){
    let canvas = document.getElementById("c");
    canvas.height = canvas.scrollHeight;
    canvas.width = canvas.scrollWidth;

    console.log(canvas)
    
    let viewer = new Viewer("c");
    let userEventHandler = new UserEventHandler(viewer, window);
    userEventHandler.initHandlers();
    let nbInstances = 200 * 200;
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag", nbInstances);

    const url = 
        'ws://'
        + window.location.host
        + '/ws/viewer/';

    let socketHandler = new SocketHandler()

    // for instance, data is an array of bool
    socketHandler.onDataReceived = function(data) {
        console.log("received")
        viewer.updateState(data);
    }


    socketHandler.connectSocket(url);

    document.querySelector('#buttonPlay').onclick = function(e) {
        if (!socketHandler.isRunning()){
            socketHandler.start(nbInstances);
            console.log("START")
        }
    }

    document.querySelector('#buttonPause').onclick = function(e) {
        if (socketHandler.isRunning()){
            socketHandler.stop();
            console.log(socketHandler)
        }
    }

    
    function loop(time){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

window.onload = function () {
    main()
}