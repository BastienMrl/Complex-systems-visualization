import { Viewer, AnimableValue } from "./viewer.js";
import { SocketHandler } from "./socketHandler.js";
import { UserInterface } from "./userInterface.js"
import { InputType, StatesTransformer, TransformType } from "./statesTransformer.js";

export var idColor;
export var transformer : StatesTransformer;
export var viewer : Viewer;

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
    
    
    viewer = new Viewer("c");
    //.... Transformer : backend data -> visualization ....
    
    transformer = new StatesTransformer();
    
    // returned id is used to update Transformer params
    // second parameter defines states used from backend data
    let idX = transformer.addTransformer(TransformType.POSITION_X, InputType.POSITION_X, 0.95);
    let idY = transformer.addTransformer(TransformType.POSITION_Z, InputType.POSITION_Y, 0.95);
    // third parameter defines here the elevation
    let idZ = transformer.addTransformer(TransformType.POSITION_Y, InputType.STATE_0, 1.5);
    const c1 = [0.0392156862745098, 0.23137254901960785, 0.28627450980392155];
    const c2 = [0.8705882352941177, 0.8901960784313725, 0.9294117647058824];
    idColor = transformer.addTransformer(TransformType.COLOR, InputType.STATE_0, c2, c1);
    
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
    await viewer.shaderProgram.updateProgramTransformers(transformer.generateTransformersBlock());
    viewer.loopAnimation();
}

window.onload = function () {
    main()
}