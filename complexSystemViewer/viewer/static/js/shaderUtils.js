async function getShaderFromFile(src, type, gl) {
    const response = await fetch(src);
    const text = await response.text();

    const shader = gl.createShader(type);
    gl.shaderSource(shader, text);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        throw `Could not compile WebGL program. \n\n${info}`;
    }

    return shader; 
}

async function initShaders(gl, srcVertex, srcFragment) { 
    const vertexShader = await getShaderFromFile(srcVertex, gl.VERTEX_SHADER, gl); 
    const fragmentShader = await getShaderFromFile(srcFragment, gl.FRAGMENT_SHADER, gl);

    let shaderProgram = gl.createProgram(); 
    gl.attachShader(shaderProgram, vertexShader); 
    gl.attachShader(shaderProgram, fragmentShader); 
    gl.linkProgram(shaderProgram); 

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) { 
        alert("Could not initialise shaders"); 
    } 

    return shaderProgram;
}

export {initShaders}