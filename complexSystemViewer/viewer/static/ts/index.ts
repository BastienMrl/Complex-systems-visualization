import { Viewer, AnimableValue } from "./viewer.js";
import { SocketHandler } from "./socketHandler.js";
import { UserInterface } from "./userInterface.js"
import { StatesTransformer, TransformType } from "./statesTransformer.js";

export var idColor;
export var transformer;

async function main(){
    let canvas : HTMLCanvasElement | null = document.getElementById("c") as HTMLCanvasElement | null;
    if (canvas == null) {
        throw "Could not find canvas";
    }
    canvas.height = canvas.clientHeight;
    canvas.width = canvas.clientWidth;
    

    
    const url = 
    'ws://'
    + window.location.host
    + '/ws/viewer/';
    
    
    let viewer = new Viewer("c");
    //.... Transformer : backend data -> visualization ....
    
    transformer = new StatesTransformer();
    
    // returned id is used to update Transformer params
    // second parameter defines states used from backend data
    let idX = transformer.addTransformer(TransformType.POSITION_X, 0, 0.95);
    let idY = transformer.addTransformer(TransformType.POSITION_Z, 1, 0.95);
    // third parameter defines here the elevation
    let idZ = transformer.addTransformer(TransformType.POSITION_Y, 2, 1.5);
    const c1 = [0.0392156862745098, 0.23137254901960785, 0.28627450980392155];
    const c2 = [0.8705882352941177, 0.8901960784313725, 0.9294117647058824];
    idColor = transformer.addTransformer(TransformType.COLOR, 2, c2, c1);
    
    viewer.setCurrentTransformer(transformer);
    
    // example, increase elevation:
    
    // transformer.setParams(idZ, 3.);
    
    //......................................................
    
    
    
    //.... AnimationCurves ....
    // default animation curve is linear
    
    // ease out expo from https://easings.net/
    let easeOut = function(time : number){ return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
    let fc0 = function(time : number){ return 1 };
    viewer.bindAnimationCurve(AnimableValue.COLOR, easeOut);
    viewer.bindAnimationCurve(AnimableValue.TRANSLATION, easeOut);
    
    
    //.........................


    // socket init before viewer init
    let socketHandler = SocketHandler.getInstance();

    // for instance, data is an array of bool
    socketHandler.onDataReceived = function(data) {
        viewer.statesBuffer.onStateReceived(data);
    }

    socketHandler.onStart = function(){
        viewer.startVisualizationAnimation();
    }

    socketHandler.onStop = function(){
        viewer.stopVisualizationAnimation();
    }
    

    await socketHandler.connectSocket(url);


    let userInterface = UserInterface.getInstance();
    userInterface.initHandlers(viewer);


    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag", userInterface.nbInstances);
    
    
    
    

    function loop(time : number){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

window.onload = function () {
    main()
}