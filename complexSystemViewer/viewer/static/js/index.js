import { Viewer } from "./viewer.js"

async function main(){
    let viewer = new Viewer("c");
    await viewer.initialization("/static/shaders/simple.vert", "/static/shaders/simple.frag");
    function loop(time){
        viewer.render(time);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}


main()