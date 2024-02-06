import { Viewer } from "./viewer.js"
import { SocketHandler } from "./socketHandler.js";

async function main(){
    let canvas = document.getElementById("c");
    canvas.height = canvas.scrollHeight;
    canvas.width = canvas.scrollWidth;

    console.log(canvas)
    
    let viewer = new Viewer("c");
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

    // camera commands 
    viewer.canvas.addEventListener('wheel', (e) =>{
        let delta = e.deltaY * 0.001;
        viewer.camera.moveForward(-delta);
    });

    
    let mousePressed = false;
    viewer.canvas.addEventListener('mousedown', (e) =>{
        if (e.button == 1)
            mousePressed = true;
    });

    viewer.canvas.addEventListener('mouseup', (e) => {
        if (e.button == 1)
            mousePressed = false;
    });

    viewer.canvas.addEventListener('mousemove', (e) => {
        if (mousePressed)
            viewer.camera.rotateCamera(e.movementY * 0.005, e.movementX * 0.005);
    })
    //....................................................


    
    function loop(time){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}


main()