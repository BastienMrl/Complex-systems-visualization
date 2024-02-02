import { Viewer } from "./viewer.js"
import { SocketHandler } from "./socketHandler.js";

async function main(){
    let viewer = new Viewer("c");
    let nbInstances = 40 * 40;
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag", nbInstances);

    const url = 
        'ws://'
        + window.location.host
        + '/ws/viewer/';

    let socketHandler = new SocketHandler()

    // for instance, data is an array of bool
    socketHandler.onDataReceived = function(data) {
        viewer.updateState(data);
    }


    socketHandler.connectSocket(url);

    document.querySelector('#button').onclick = function(e) {
        if (socketHandler.isRunning()){
            socketHandler.stop();
            document.querySelector('#button').value = "Start";
        }
        else{
            document.querySelector('#button').value = "Stop"
            socketHandler.start(nbInstances);
        }
    }
    
    
    function loop(time){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}


main()