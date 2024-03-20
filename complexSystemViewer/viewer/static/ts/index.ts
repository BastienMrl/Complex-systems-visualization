import { Viewer } from "./viewer.js";
import { UserInterface } from "./interface/userInterface.js"


async function main(){
    let canvas : HTMLCanvasElement | null = document.getElementById("c") as HTMLCanvasElement | null;
    if (canvas == null) {
        throw "Could not find canvas";
    }
    canvas.height = canvas.clientHeight;
    canvas.width = canvas.clientWidth;
    
    let viewer = new Viewer("c");

    let userInterface = UserInterface.getInstance();
    userInterface.initHandlers(viewer);
    
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag");
    viewer.loopAnimation();
}

window.onload = function () {
    main()
}