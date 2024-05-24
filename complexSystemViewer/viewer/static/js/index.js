import { ViewerManager, ViewerType } from "./viewerManager.js";
import { UserInterface } from "./interface/userInterface.js";
async function main() {
    let canvas = document.getElementById("c");
    if (canvas == null) {
        throw "Could not find canvas";
    }
    canvas.height = canvas.clientHeight;
    canvas.width = canvas.clientWidth;
    let viewer = new ViewerManager("c");
    let userInterface = UserInterface.getInstance();
    userInterface.initHandlers(viewer);
    await viewer.initialization(ViewerType.MULTIPLE_MESHES);
    viewer.loopAnimation();
}
window.onload = function () {
    main();
};
