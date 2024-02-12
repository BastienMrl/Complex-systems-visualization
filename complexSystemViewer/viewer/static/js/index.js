import { Viewer, AnimableValue } from "./viewer.js";
import { SocketHandler } from "./socketHandler.js";
import { UserEventHandler } from "./userEventHandler.js";
import { StatesTransformer, TransformType } from "./statesTransformer.js";
async function main() {
    let canvas = document.getElementById("c");
    if (canvas == null) {
        throw "Could not find canvas";
    }
    canvas.height = canvas.clientHeight;
    canvas.width = canvas.clientWidth;
    // socket init before viewer init
    let socketHandler = SocketHandler.getInstance();
    // for instance, data is an array of bool
    socketHandler.onDataReceived = function (data) {
        viewer.statesBuffer.onStateReceived(data);
    };
    socketHandler.onStart = function () {
        viewer.startVisualizationAnimation();
    };
    socketHandler.onStop = function () {
        viewer.stopVisualizationAnimation();
    };
    const url = 'ws://'
        + window.location.host
        + '/ws/viewer/';
    await socketHandler.connectSocket(url);
    let viewer = new Viewer("c");
    let userEventHandler = UserEventHandler.getInstance();
    userEventHandler.initHandlers(viewer);
    let nbInstances = 100 * 100;
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag", nbInstances);
    //.... Transformer : backend data -> visualization ....
    let transformer = new StatesTransformer();
    // returned id is used to update Transformer params
    // second parameter defines states used from backend data
    let idX = transformer.addTransformer(TransformType.POSITION_X, 0, 0.95);
    let idY = transformer.addTransformer(TransformType.POSITION_Z, 1, 0.95);
    // third parameter defines here the elevation
    let idZ = transformer.addTransformer(TransformType.POSITION_Y, 2, 1.5);
    const c1 = [0.0392156862745098, 0.23137254901960785, 0.28627450980392155];
    const c2 = [0.8705882352941177, 0.8901960784313725, 0.9294117647058824];
    let idColor = transformer.addTransformer(TransformType.COLOR, 2, c2, c1);
    viewer.setCurrentTransformer(transformer);
    // example, increase elevation:
    // transformer.setParams(idZ, 3.);
    //......................................................
    //.... AnimationCurves ....
    // default animation curve is linear
    // ease out expo from https://easings.net/
    let easeOut = function (time) { return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
    let fc0 = function (time) { return 1; };
    viewer.bindAnimationCurve(AnimableValue.COLOR, easeOut);
    viewer.bindAnimationCurve(AnimableValue.TRANSLATION, easeOut);
    //.........................
    document.querySelector('#buttonPlay').onclick = function (e) {
        if (!socketHandler.isRunning) {
            socketHandler.start(nbInstances);
        }
    };
    document.querySelector('#buttonPause').onclick = function (e) {
        if (socketHandler.isRunning) {
            socketHandler.stop();
        }
    };
    function loop(time) {
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}
window.onload = function () {
    main();
};
