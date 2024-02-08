async function getShaderFromFile(src : string, type : number, gl : WebGL2RenderingContext) : Promise<WebGLShader> {
    const response : Response = await fetch(src);
    const text : string = await response.text();

    const shader : WebGLShader | null = gl.createShader(type);
    if (shader == null){
        throw "Could not create Shader from source files";
    }
    gl.shaderSource(shader, text);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        throw `Could not compile WebGL program. \n\n${info}`;
    }

    return shader; 
}

async function initShaders(gl : WebGL2RenderingContext, srcVertex : string, srcFragment : string) : Promise<WebGLProgram>{ 
    const vertexShader : WebGLShader = await getShaderFromFile(srcVertex, gl.VERTEX_SHADER, gl); 
    const fragmentShader : WebGLShader = await getShaderFromFile(srcFragment, gl.FRAGMENT_SHADER, gl);

    let shaderProgram : WebGLProgram | null = gl.createProgram(); 
    if (shaderProgram == null){
        throw "Cour not create Shader Program";
    }
    gl.attachShader(shaderProgram, vertexShader); 
    gl.attachShader(shaderProgram, fragmentShader); 
    gl.linkProgram(shaderProgram); 

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) { 
        alert("Could not initialise shaders"); 
    } 

    return shaderProgram;
}

export {initShaders}